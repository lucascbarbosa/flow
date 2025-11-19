"""Training utilities."""
import torch
import torch.nn as nn
from src.models.cnf import CNF
from src.models.neural_ode import NeuralODE
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional


def train_neural_ode(
    model: NeuralODE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100
) -> None:
    """Train Neural ODE to learn transformation from N(0, I) to data.

    The model learns to transform samples from a standard normal distribution
    N(0, I) to match the target data distribution.

    Args:
        model (NeuralODE): NeuralODE model.

        dataloader (DataLoader): DataLoader for training data.

        optimizer (Optimizer): Optimizer for training.

        device (Device): Device to run training on.

        num_epochs (int): Number of training epochs.
    """
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x_target = batch.to(device)  # Target data
            batch_size, n_features = x_target.shape

            optimizer.zero_grad()

            # Sample initial conditions from Gaussian distribution N(0, I)
            x0 = torch.randn(batch_size, n_features, device=device)

            # Forward: integrate from t=0 to t=1
            x_t = model(x0)
            z = x_t[-1]  # Final state after transformation

            # Loss: MSE between transformed samples and target data
            loss = nn.functional.mse_loss(z, x_target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")


def train_cnf(
    model: CNF,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100
) -> None:
    """Train CNF using negative log-likelihood.

    Args:
        model (CNF): CNF model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for training.
        device (Device): Device to run training on.
        num_epochs (int): Number of training epochs.
    """
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_nll = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x = batch.to(device)

            optimizer.zero_grad()

            # Calculate log-likelihood
            log_prob = model.log_prob(x)

            # Loss: negative log-likelihood
            loss = -log_prob.mean()

            loss.backward()

            # Gradient clipping (optional, but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            total_nll += -log_prob.mean().item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_nll = total_nll / n_batches
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, NLL: {avg_nll:.4f}")


def count_nfe(
    model: NeuralODE | CNF,
    x: torch.Tensor,
    t_span: Optional[torch.Tensor] = None
) -> int:
    """Count number of function evaluations (NFEs) during integration.

    Args:
        model (NeuralODE | CNF): NeuralODE or CNF model with vf attribute.
        x (torch.Tensor): Input tensor.
        t_span (torch.Tensor): Time points for integration.

    Returns:
        int: Number of function evaluations.
    """
    nfe = [0]

    def count_wrapper(t: torch.Tensor, x_state: torch.Tensor) -> torch.Tensor:
        nfe[0] += 1
        return model.vf(t, x_state)

    # Create temporary wrapper
    original_vf = model.vf
    model.vf = count_wrapper

    # Forward pass
    with torch.no_grad():
        _ = model(x, t_span)

    # Restore
    model.vf = original_vf

    return nfe[0]
