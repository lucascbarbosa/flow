"""Continuous Normalizing Flow (CNF) implementation."""
import torch
import torch.nn as nn
from src.models.vector_field import VectorField
from src.utils.trace import divergence_exact
from torchdiffeq import odeint_adjoint
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
            features = vector_field.features
            self.base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(features, device=device),
                torch.eye(features, device=device)
            )
        else:
            self.base_dist = base_dist

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

        x = x.requires_grad_(True)

        # Compute vector field
        dx_dt = self.vf(t, x)

        # Compute trace of the Jacobian using exact method
        with torch.enable_grad():
            trace = divergence_exact(lambda x_: self.vf(t, x_), x)

        dlogdet_dt = -trace.unsqueeze(-1)  # (batch, 1)

        return torch.cat([dx_dt, dlogdet_dt], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor | None = None,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform x -> z (forward) or z -> x (reverse).

        Args:
            x (torch.Tensor): Input tensor with shape (batch, features).
            t_span (torch.Tensor): Time points to evaluate.
            reverse (bool): If True, integrates from t=1 to t=0 (z -> x).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - z (torch.Tensor): Transformed tensor with shape
                    (batch, features).
                - log_det (torch.Tensor): Log determinant with shape
                    (batch, 1).
        """
        if t_span is None:
            if reverse:
                # z -> x: integrate from t=1 to t=0
                t_span = torch.tensor([1., 0.], device=device, dtype=x.dtype)
            else:
                # x -> z: integrate from t=0 to t=1
                t_span = torch.tensor([0., 1.], device=device, dtype=x.dtype)

        # Ensure x is 2D: [batch, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Ensure x is on the correct device
        x = x.to(device)

        log_det_init = torch.zeros(
            x.shape[0], 1,
            device=device,
            dtype=x.dtype,
        )
        state_init = torch.cat([x, log_det_init], dim=-1)

        # Ensure t_span is on the correct device
        t_span = t_span.to(device)

        # odeint_adjoint handles both input and parameter gradients
        # automatically. No need to specify adjoint_params.
        state_t = odeint_adjoint(
            self._augmented_dynamics,
            state_init,
            t_span,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            adjoint_params=tuple(self.vf.parameters())
        )

        # Final state
        state_final = state_t[-1]  # (batch, features + 1)
        z = state_final[:, :-1]  # (batch, features)
        log_det = state_final[:, -1]  # (batch,)

        return z, log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate log p(x) using change of variables.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, features).

        Returns:
            torch.Tensor: Log probability with shape (batch,).
        """
        if torch.is_grad_enabled() and not x.requires_grad:
            x = x.requires_grad_(True)

        # Transform x -> z
        z, log_det = self.forward(x, reverse=False)

        # log p(x) = log p(z) + log |det(∂z/∂x)|
        # Since we integrate from x to z, log_det is log |det(∂z/∂x)|
        log_prob_z = self.base_dist.log_prob(z)
        log_prob_x = log_prob_z + log_det

        return log_prob_x

    def sample(self, num_samples: int) -> torch.Tensor:
        """Generate samples x ~ p(x) via z ~ p(z) -> x.

        Args:
            num_samples (int): Number of samples.

        Returns:
            torch.Tensor: Samples with shape (num_samples, features).
        """
        # Ensure base_dist is on the same device as model parameters
        device = next(self.vf.parameters()).device
        if self.base_dist.loc.device != device:
            # Recreate base_dist on the correct device
            features = self.vf.features
            self.base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(features, device=device),
                torch.eye(features, device=device)
            )

        # Sample z ~ p(z)
        z = self.base_dist.sample((num_samples,))

        # Transform z -> x
        x, _ = self.forward(z, reverse=True)

        return x
