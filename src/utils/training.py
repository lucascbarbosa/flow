"""Training utilities."""
import torch
import torch.nn as nn
from src.models.cnf import CNF
from src.models.neural_ode import NeuralODE
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional


def rbf_kernel(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """Compute RBF (Gaussian) kernel matrix k(x, y) = exp(-γ ||x - y||²).

    Args:
        x (torch.Tensor): First set of samples with shape (n, d).
        y (torch.Tensor): Second set of samples with shape (m, d).
        gamma (float): Kernel bandwidth parameter.

    Returns:
        torch.Tensor: Kernel matrix with shape (n, m).
    """
    # Compute pairwise squared distances
    # ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2 * x_i^T y_j
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (n, 1)
    y_norm = (y ** 2).sum(dim=1, keepdim=True).T  # (1, m)
    xy = torch.mm(x, y.T)  # (n, m)

    squared_dist = x_norm + y_norm - 2 * xy
    return torch.exp(-gamma * squared_dist)


def mmd2_loss(
    x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """Compute MMD² (Maximum Mean Discrepancy squared).

    MMD²(X,Y) = (1/n²) Σᵢⱼ k(xᵢ, xⱼ) + (1/m²) Σᵢⱼ k(yᵢ, yⱼ)
                - (2/nm) Σᵢⱼ k(xᵢ, yⱼ)

    Args:
        x (torch.Tensor): First set of samples with shape (n, d).
        y (torch.Tensor): Second set of samples with shape (m, d).
        gamma (float): RBF kernel bandwidth parameter.

    Returns:
        torch.Tensor: MMD² value (scalar).
    """
    n = x.shape[0]
    m = y.shape[0]

    # Compute kernel matrices
    k_xx = rbf_kernel(x, x, gamma)  # (n, n)
    k_yy = rbf_kernel(y, y, gamma)  # (m, m)
    k_xy = rbf_kernel(x, y, gamma)  # (n, m)

    # MMD² = (1/n²) Σᵢⱼ k(xᵢ, xⱼ) + (1/m²) Σᵢⱼ k(yᵢ, yⱼ)
    #        - (2/nm) Σᵢⱼ k(xᵢ, yⱼ)
    term1 = k_xx.sum() / (n * n)
    term2 = k_yy.sum() / (m * m)
    term3 = k_xy.sum() * 2 / (n * m)

    mmd2 = term1 + term2 - term3
    return mmd2


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
            z = torch.randn(batch_size, n_features, device=device)

            # Forward: integrate from t=0 to t=1
            x_t = model(z)
            x = x_t[-1]  # Final state after transformation

            # Loss: MMD² between transformed samples and target data
            loss = mmd2_loss(x, x_target)

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
    with torch.no_grad():
        _ = model(x, t_span)

    # Get count
    nfe_count = counting_vf.nfe

    # Restore original vector field
    model.vf = original_vf

    return nfe_count
