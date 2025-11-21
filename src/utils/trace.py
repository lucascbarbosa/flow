"""Trace computation utilities for CNF."""
import torch
from typing import Callable, Literal
from torch import autograd


def divergence_exact(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor
) -> torch.Tensor:
    """Calculate trace(∂f/∂x) exactly using autograd.

    WARNING: Cost O(d²) - only viable for low dimensions!

    Args:
        f (Callable[[torch.Tensor], torch.Tensor]): Function R^d -> R^d
            that accepts x and returns (batch, d).

        x (torch.Tensor): Input tensor with shape (batch, d).

    Returns:
        torch.Tensor: Trace of the Jacobian with shape (batch,).
    """
    batch_size, dim = x.shape

    # Enable gradients
    x = x.requires_grad_(True)

    # Compute f(x)
    f_x = f(x)  # (batch, dim)

    # For each dimension i, calculate ∂f_i/∂x_i
    trace = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

    for i in range(dim):
        # Compute ∂f[:, i]/∂x
        df_i = autograd.grad(
            f_x[:, i].sum(),
            x,
            create_graph=True,
            retain_graph=True
        )[0]  # (batch, dim)

        # Sum only the diagonal: ∂f_i/∂x_i
        trace += df_i[:, i]

    return trace


def divergence_hutchinson(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    num_samples: int = 1,
    distribution: Literal['rademacher', 'gaussian'] = 'rademacher'
) -> torch.Tensor:
    """Calculate trace(∂f/∂x) using Hutchinson estimator.

    Cost O(d) - scalable for high dimensions!

    Args:
        f: Function R^d -> R^d (callable that accepts x and returns
            (batch, d)).

        x: Input tensor with shape (batch, d).

        num_samples: Number of samples for estimation (default: 1).

        distribution: 'rademacher' or 'gaussian'.

    Returns:
        trace: Trace estimate with shape (batch,).
    """
    batch_size, dim = x.shape

    # Enable gradients
    x = x.requires_grad_(True)

    # Compute f(x)
    f_x = f(x)  # (batch, dim)

    trace_estimates = []

    for _ in range(num_samples):
        # Sample noise vector
        if distribution == 'rademacher':
            # Rademacher: ε ~ Uniform({-1, +1})
            epsilon = (
                torch.randint(0, 2, (batch_size, dim), device=x.device,
                              dtype=x.dtype) * 2 - 1
            )
        elif distribution == 'gaussian':
            # Gaussian: ε ~ N(0, I)
            epsilon = torch.randn(
                batch_size, dim, device=x.device, dtype=x.dtype
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Compute ε^T * f(x)
        vTf = (epsilon * f_x).sum(dim=-1)  # (batch,)

        # Compute gradient: ∂(ε^T * f(x))/∂x = ε^T * ∂f/∂x
        grad_vTf = autograd.grad(
            vTf.sum(),
            x,
            create_graph=True,
            retain_graph=True
        )[0]  # (batch, dim)

        # Hutchinson estimator: ε^T * (∂f/∂x) * ε
        trace_est = (epsilon * grad_vTf).sum(dim=-1)  # (batch,)
        trace_estimates.append(trace_est)

    # Average over samples
    trace = torch.stack(trace_estimates, dim=0).mean(dim=0)  # (batch,)

    return trace
