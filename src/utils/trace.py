"""Trace computation utilities for CNF."""
import torch
from typing import Callable, Literal
from torch import autograd


def divergence_exact(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """Calculate trace(∂f/∂x) exactly using autograd.

    WARNING: Cost O(d²) - only viable for low dimensions!
    For high dimensions (d > 50), consider using divergence_hutchinson.

    Optimized implementation: computes the full Jacobian at once using
    batched gradients, then extracts the trace via einsum.

    Args:
        f (Callable[[torch.Tensor], torch.Tensor]): Function R^d -> R^d
            that accepts x and returns (batch, d).

        x (torch.Tensor): Input tensor with shape (batch, d).

    Returns:
        torch.Tensor: Trace of the Jacobian with shape (batch,).
    """
    # Create identity matrix for batched gradient computation
    identity = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    grad_outputs = identity.expand(*x.shape, -1).movedim(-1, 0)
    # grad_outputs shape: (dim, batch, dim)

    if not x.requires_grad:
        x = x.clone().requires_grad_(True)

    f_x = f(x)  # (batch, dim)

    # Compute full Jacobian using batched gradients
    (jacobian,) = torch.autograd.grad(
        f_x,
        x,
        grad_outputs,
        create_graph=True,
        is_grads_batched=True
    )  # (dim, batch, dim)

    # Extract trace: sum over diagonal elements
    trace = torch.einsum("i...i", jacobian)  # (batch,)
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

    # Enable gradients if not already enabled
    # Use clone if x doesn't require grad to ensure proper graph connection
    if not x.requires_grad:
        x = x.clone().requires_grad_(True)
    else:
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
