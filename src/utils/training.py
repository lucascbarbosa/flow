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
    """Train Neural ODE for classification task.

    The model integrates the ODE and classifies the final state.
    Returns both trajectory and logits, but only uses logits for training.

    Args:
        model (NeuralODE): NeuralODE model with classification head.

        dataloader (DataLoader): DataLoader for training data.

        optimizer (Optimizer): Optimizer for training.

        device (Device): Device to run training on.

        num_epochs (int): Number of training epochs.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total_samples = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x_data, labels = batch
            x_data = x_data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Use data as initial condition
            z = x_data

            # Forward: integrate ODE and classify
            # Returns trajectory and logits (trajectory kept for visualization)
            _, logits = model(z)

            # Loss: Cross-entropy
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        accuracy = 100 * correct / total_samples
        print(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}, "
            f"Accuracy: {accuracy:.2f}%"
        )

    return avg_loss


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
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            x = batch[0].to(device)
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
    # Create counting wrapper module
    counting_vf = CountingVectorField(model.vf)

    # Temporarily replace the vector field
    original_vf = model.vf
    model.vf = counting_vf

    # Forward pass
    # CNF forward takes (x, reverse), NeuralODE takes (x, t_span)
    with torch.no_grad():
        if isinstance(model, CNF):
            _, _ = model(x, reverse=False)
        else:
            _, _ = model(x, t_span)

    # Get count
    nfe_count = counting_vf.nfe

    # Restore original vector field
    model.vf = original_vf

    return nfe_count
