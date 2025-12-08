"""Training utilities."""
import torch
import torch.nn as nn
from src.models.cnf import CNF
from src.models.neural_ode import NeuralODE
from src.models.ffjord import FFJORD
from torch.utils.data import DataLoader
from tqdm import tqdm
from zuko.flows import RealNVP


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_neural_ode(
    model: NeuralODE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 100,
    n_steps: int = 100,
    n_samples: int = 25,
) -> None:
    """Train Neural ODE using Maximum Mean Discrepancy (MMD) loss.

    Uses four loss components:
    1. Forward loss: ensures data → Gaussian transformation (MMD)
    2. Backward loss: ensures Gaussian → data transformation (MMD)
    3. Reversibility loss: ensures forward→backward returns to original (MSE)

    Args:
        model: Neural ODE model.
        dataloader: DataLoader for training data.
        optimizer: Optimizer for training.
        n_epochs: Number of training epochs.
        n_steps: Number of steps to integrate the ODE.
        n_samples: Number of samples to use for MMD loss.
    """
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        for x0_target in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}"):
            optimizer.zero_grad()
            batch_size = x0_target.shape[0]
            features = x0_target.shape[1]

            # 1. Forward trajectory: x0_target → x1_pred
            x0_target = x0_target.to(device)
            forward_x_t = model(x0_target, n_steps)
            forward_x_t = forward_x_t.permute(1, 2, 0)

            # 2. Backward trajectory: x1_target → x0_pred
            x1_target = torch.randn_like(x0_target)
            backward_x_t = model.backward(x1_target, n_steps)
            backward_x_t = torch.flip(backward_x_t, dims=[0])
            backward_x_t = backward_x_t.permute(1, 2, 0)

            # Sample n_samples random time points for each trajectory in batch
            # time_indices = torch.randint(
            #     0, n_steps,
            #     (batch_size, n_samples),
            #     device=x0_target.device
            # )
            time_indices = torch.linspace(
                0, n_steps - 1, n_samples,
                dtype=torch.long,
                device=x0_target.device
            ).unsqueeze(0).expand(batch_size, n_samples)

            # Use advanced indexing to sample from each trajectory
            batch_indices = torch.arange(
                batch_size, device=x0_target.device
            ).unsqueeze(1)

            # Sample forward states: (batch_size, features, n_samples)
            sample_forward = forward_x_t[
                batch_indices, :, time_indices
            ]
            sample_forward = sample_forward.permute(0, 2, 1)
            sample_forward = sample_forward.reshape(-1, features)

            # Sample backward states: (batch_size, features, n_samples)
            sample_backward = backward_x_t[
                batch_indices, :, time_indices
            ]
            sample_backward = sample_backward.permute(0, 2, 1)
            sample_backward = sample_backward.reshape(-1, features)

            loss = ((sample_forward - sample_backward) ** 2).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {avg_loss:.6f} "
        )

    return avg_loss


def train_cnf(
    model: CNF,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 100,
) -> None:
    """Train CNF using negative log-likelihood."""
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        for x0 in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}"):
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
    flow: RealNVP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 100
) -> None:
    """Train RealNVP flow using negative log-likelihood.

    Args:
        flow (RealNVP): Zuko RealNVP flow model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for training.
        device (Device): Device to run training on.
        n_epochs (int): Number of training epochs.
    """
    flow.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0

        for x0 in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}"
        ):
            # Extract data from batch (handle both single-tensor and tuple)
            x = x0.to(device).to(torch.float32)

            # RealNVP models in this codebase don't use context, pass None
            optimizer.zero_grad()

            # Calculate log-likelihood
            log_prob = flow(None).log_prob(x)

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


def train_ffjord(
    model: FFJORD,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 100,
    lambda_ke: float = 0.01,
    warmup_epochs: int = 5
) -> None:
    """Train FFJORD using NLL with regularization.

    Args:
        model (FFJORD): FFJORD model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for training.
        device (Device): Device to run training on.
        n_epochs (int): Number of training epochs.
        lambda_ke (float): Weight for kinetic energy regularization.
            Default is 0.01.
        warmup_epochs (int): Number of epochs for linear learning rate warmup.
            Default is 5.
    """
    model.train()
    initial_lr = optimizer.param_groups[0]['lr']

    for epoch in range(n_epochs):
        # Linear warmup for learning rate
        if epoch < warmup_epochs:
            lr = initial_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        total_loss = 0.0
        total_nll = 0.0
        total_ke = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}"):
            x = batch[0].to(device)

            optimizer.zero_grad()

            # Calculate log-likelihood
            log_prob = model.log_prob(x)
            nll = -log_prob.mean()

            # Optional: Kinetic energy regularization
            # KE = 0.5 * ||f(x, t)||^2
            ke_loss = 0.0
            if lambda_ke > 0:
                # Sample random time points
                t = torch.rand(x.shape[0], device=device)
                x_requires_grad = x.requires_grad_(True)

                # Compute vector field
                dx_dt = model.vf(t, x_requires_grad)

                # Kinetic energy: 0.5 * ||dx_dt||^2
                ke_loss = 0.5 * (dx_dt ** 2).sum(dim=-1).mean()

            # Total loss
            loss = nll + lambda_ke * ke_loss

            loss.backward()

            # Gradient clipping (recommended for FFJORD)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            total_nll += nll.item()
            total_ke += ke_loss.item() if lambda_ke > 0 else 0.0
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_nll = total_nll / n_batches
        avg_ke = total_ke / n_batches if lambda_ke > 0 else 0.0

        if lambda_ke > 0:
            print(
                f"Epoch {epoch + 1}, "
                f"Loss: {avg_loss:.4f}, "
                f"NLL: {avg_nll:.4f}, "
                f"KE: {avg_ke:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}, "
                f"Loss: {avg_loss:.4f}, "
                f"NLL: {avg_nll:.4f}"
            )


class CountingVectorField2D(nn.Module):
    """Wrapper module that counts function evaluations."""
    def __init__(self, vf: nn.Module):
        """."""
        super().__init__()
        self.vf = vf
        self.nfe = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        self.nfe += 1
        return self.vf(t, x)


def count_nfe(
    model: NeuralODE | CNF,
    x: torch.Tensor,
    n_steps: int = 10
) -> int:
    """Count number of function evaluations (NFEs) during integration."""
    # Create counting wrapper module
    counting_vf = CountingVectorField2D(model.vf)

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
