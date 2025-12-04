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
    num_epochs: int = 100,
    n_steps: int = 100
) -> None:
    """Train Neural ODE for classification task."""
    model.train()
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0

        for x0 in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            x0 = x0.to(device)

            # Forward: integrate ODE
            x_t = model(x0, n_steps)
            z = x_t[-1]

            # Generate random target
            z_target = torch.randn_like(z)

            # Loss: Mean Squared Error
            loss = criterion(z, z_target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")

    return avg_loss


def train_cnf(
    model: CNF,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    n_steps: int = 100
) -> None:
    """Train CNF using negative log-likelihood.

    Args:
        model (CNF): CNF model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for training.
        device (Device): Device to run training on.
        num_epochs (int): Number of training epochs.
        n_steps (int): Number of steps to integrate the ODE.
    """
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0

        for x0 in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            x0 = x0.to(device)

            # Calculate log-likelihood
            log_prob = model.log_prob(x0)

            # Loss: negative log-likelihood
            loss = -log_prob.mean()

            loss.backward()

            # Gradient clipping (optional, but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")


def train_realnvp(
    flow,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100
) -> None:
    """Train RealNVP flow using negative log-likelihood.

    Args:
        flow: Zuko RealNVP flow model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for training.
        device (Device): Device to run training on.
        num_epochs (int): Number of training epochs.
    """
    flow.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x = batch[0].to(device)
            c = batch[1].to(device)
            optimizer.zero_grad()

            # Calculate log-likelihood
            log_prob = flow(c).log_prob(x)

            # Loss: negative log-likelihood
            loss = -log_prob.mean()

            loss.backward()

            # Gradient clipping (optional, but recommended)
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")


class CountingVectorField(nn.Module):
    """Wrapper module that counts function evaluations."""
    def __init__(self, vf: nn.Module):
        super().__init__()
        self.vf = vf
        self.nfe = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        return self.vf(t, x)


def count_nfe(
    model: NeuralODE | CNF,
    x: torch.Tensor,
    n_steps: int = 100
) -> int:
    """Count number of function evaluations (NFEs) during integration."""
    # Create counting wrapper module
    counting_vf = CountingVectorField(model.vf)

    # Temporarily replace the vector field
    original_vf = model.vf
    model.vf = counting_vf

    # Forward pass
    with torch.no_grad():
        model(x, n_steps)

    # Get count
    nfe_count = counting_vf.nfe

    # Restore original vector field
    model.vf = original_vf

    return nfe_count
