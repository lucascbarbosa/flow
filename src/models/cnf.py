"""Continuous Normalizing Flow (CNF) implementation."""
import torch
import torch.nn as nn
from src.models.vector_field import VectorField
from src.utils.trace import divergence_exact
from torchdiffeq import odeint
from torch.distributions import Distribution
from typing import Literal, Optional, Tuple


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNF(nn.Module):
    """Continuous Normalizing Flow with exact trace computation."""
    def __init__(
        self,
        vector_field: VectorField,
        method: Literal['dopri5', 'euler', 'rk4'] = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4,
        base_dist: Optional[Distribution] = None
    ) -> None:
        """Initialize CNF.

        Args:
            vector_field (VectorField): Vector field module f(x, t).

            method (Literal['dopri5', 'euler', 'rk4']): ODE solver method.
                Default is 'dopri5'.

            rtol (float): Relative tolerance for ODE solver. Default is 1e-3.

            atol (float): Absolute tolerance for ODE solver. Default is 1e-4.

            base_dist (Distribution, optional): Base distribution.
                If None, uses N(0, I).
        """
        super().__init__()
        self.vf = vector_field
        self.method = method
        self.rtol = rtol
        self.atol = atol
        if base_dist is None:
            # Prior: N(0, I)
            # Store parameters separately so they can be moved to device
            features = vector_field.features
            self.register_buffer('_base_loc', torch.zeros(features))
            self.register_buffer('_base_cov', torch.eye(features))
            self.base_dist = None
        else:
            self.base_dist = base_dist
            self._base_loc = None
            self._base_cov = None

    def _get_base_dist(self, device: torch.device) -> Distribution:
        """Get base distribution on the specified device.

        Args:
            device: Device to place distribution on.

        Returns:
            Base distribution on the specified device.
        """
        # Default: N(0, I) on correct device
        return torch.distributions.MultivariateNormal(
            self._base_loc.to(device),
            self._base_cov.to(device)
        )

    def _augmented_dynamics(
        self,
        t: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Augmented ODE: integrates [x, log_det] simultaneously.

        dx/dt = f(x, t)
        d(log_det)/dt = -trace(∂f/∂x)

        Args:
            t (torch.Tensor): Scalar time.

            state (torch.Tensor): State tensor with shape (batch, features + 1)
                containing [x, log_det].

        Returns:
            torch.Tensor: Derivative with shape (batch, features + 1).
        """
        x = state[:, :-1]  # (batch, features)
        if not x.requires_grad:
            # 1. Create a detached leaf that tracks gradients
            x_detached = x.detach()
            x_detached.requires_grad_(True)

            # 2. Compute trace within enable_grad context
            with torch.enable_grad():
                trace = divergence_exact(lambda x_: self.vf(t, x_), x_detached)

            # 3. Compute vector field normally
            dx_dt = self.vf(t, x)

        else:
            dx_dt = self.vf(t, x)
            trace = divergence_exact(lambda x_: self.vf(t, x_), x)

        dlogdet_dt = -trace.unsqueeze(-1)  # (batch, 1)

        return torch.cat([dx_dt, dlogdet_dt], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform x -> z (forward) or z -> x (reverse)."""
        if t_span is None:
            t_span = torch.tensor([0., 1.], device=x.device, dtype=x.dtype)
        else:
            t_span = t_span.to(x.device)

        if not x.requires_grad and torch.is_grad_enabled():
            x = x.clone().requires_grad_(True)

        log_det_init = torch.zeros(
            x.shape[0], 1,
            device=x.device,
            dtype=x.dtype,
        )
        state_init = torch.cat([x, log_det_init], dim=-1)

        # Integrate ODE forward
        state_t = odeint(
            self._augmented_dynamics,
            state_init,
            t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol
        )

        # Final state
        state_final = state_t[-1]  # (batch, features + 1)
        z = state_final[:, :-1]  # (batch, features)
        log_det = state_final[:, -1]  # (batch,)

        return z, log_det

    def log_prob(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculate log p(x) using change of variables."""
        # Ensure x requires grad for odeint_adjoint to work properly
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

        # Transform x -> z
        z, log_det = self.forward(x, t_span)

        # log p(x) = log p(z) + log |det(∂z/∂x)|
        # Since we integrate from x to z, log_det is log |det(∂z/∂x)|
        base_dist = self._get_base_dist(z.device)
        log_prob_z = base_dist.log_prob(z)
        log_prob_x = log_prob_z + log_det
        return log_prob_x

    def sample(
        self,
        num_samples: int,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """Generate samples x ~ p(x) via z ~ p(z) -> x."""
        t_span = torch.linspace(
            1., 0.,
            n_steps,
            device=device,
            dtype=torch.float64,
        )

        # Sample z ~ p(z)
        base_dist = self._get_base_dist(device)
        z = base_dist.sample((num_samples,)).to(device)

        # Transform z -> x
        x, _ = self.forward(z, t_span)
        return x
