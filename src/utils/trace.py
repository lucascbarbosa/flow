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
    For high dimensions (d > 50), consider using divergence_hutchinson.

    Optimized implementation: computes f(x) once, then for each output
    dimension j, computes ∂f_j/∂x and extracts the j-th component (diagonal).

    Args:
        f (Callable[[torch.Tensor], torch.Tensor]): Function R^d -> R^d
            that accepts x and returns (batch, d).

        x (torch.Tensor): Input tensor with shape (batch, d).

    Returns:
        torch.Tensor: Trace of the Jacobian with shape (batch,).
    """
    batch_size, dim = x.shape

    # Ensure x requires grad
    x = x.requires_grad_(True)

    # Compute f(x) once - this will be reused
    f_x = f(x)  # (batch, dim)

    # Compute trace by summing diagonal elements
    # For each output dimension j, compute ∂f_j/∂x and take j-th component
    trace = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

    for j in range(dim):
        # Compute gradient of f_j with respect to x
        # Using sum() to aggregate over batch for gradient computation
        grad_fj = autograd.grad(
            f_x[:, j].sum(),  # Sum over batch
            x,
            create_graph=True,
            retain_graph=True
        )[0]  # (batch, dim)

        # Extract j-th component (diagonal element): ∂f_j/∂x_j
        trace += grad_fj[:, j]  # (batch,)

    return trace


def divergence(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    method: Literal['exact', 'hutchinson', 'auto'] = 'auto',
    num_samples: int = 1,
    distribution: Literal['rademacher', 'gaussian'] = 'rademacher',
    exact_threshold: int = 50
) -> torch.Tensor:
    """Calculate trace(∂f/∂x) with automatic method selection.

    Automatically chooses between exact and Hutchinson estimator based on
    input dimension. For dimensions <= exact_threshold, uses exact method.
    For higher dimensions, uses Hutchinson estimator for better performance.

    Args:
        f: Function R^d -> R^d (callable that accepts x and returns
            (batch, d)).
        x: Input tensor with shape (batch, d).
        method: 'exact', 'hutchinson', or 'auto' (default: 'auto').
        num_samples: Number of samples for Hutchinson estimator (default: 1).
        distribution: 'rademacher' or 'gaussian' for Hutchinson.
        exact_threshold: Dimension threshold for automatic method selection
            (default: 50).

    Returns:
        trace: Trace of the Jacobian with shape (batch,).
    """
    _, dim = x.shape

    if method == 'auto':
        if dim <= exact_threshold:
            method = 'exact'
        else:
            method = 'hutchinson'

    if method == 'exact':
        return divergence_exact(f, x)
    elif method == 'hutchinson':
        return divergence_hutchinson(f, x, num_samples, distribution)
    else:
        raise ValueError(f"Unknown method: {method}")


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
