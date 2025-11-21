"""Continuous Normalizing Flow (CNF) implementation."""
import torch
import torch.nn as nn
from src.models.vector_field import VectorField
from src.utils.trace import divergence_exact
from torchdiffeq import odeint_adjoint
from torch.distributions import Distribution
from typing import Literal, Optional, Tuple


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
                torch.zeros(features),
                torch.eye(features)
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

        # Enable gradients for x
        x = x.requires_grad_(True)

        # Compute vector field
        dx_dt = self.vf(t, x)  # (batch, features)

        # Compute trace of Jacobian
        trace = divergence_exact(lambda x: self.vf(t, x), x)  # (batch,)

        # d(log_det)/dt = -trace (note the sign!)
        dlogdet_dt = -trace.unsqueeze(-1)  # (batch, 1)

        return torch.cat([dx_dt, dlogdet_dt], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform x -> z (forward) or z -> x (reverse).

        Args:
            x (torch.Tensor): Input tensor with shape (batch, features).

            reverse (bool): If True, integrates from t=1 to t=0 (z -> x).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - z (torch.Tensor): Transformed tensor with shape
                    (batch, features).
                - log_det (torch.Tensor): Log determinant with shape
                    (batch, 1).
        """
        if reverse:
            # z -> x: integrate from t=1 to t=0
            t_span = torch.tensor([1., 0.], device=x.device, dtype=x.dtype)
        else:
            # x -> z: integrate from t=0 to t=1
            t_span = torch.tensor([0., 1.], device=x.device, dtype=x.dtype)

        # Initial state: [x, log_det=0]
        # When using adjoint_params, we need state_init to require grad
        # to build the computation graph for gradient computation
        # w.r.t. parameters
        if x.requires_grad:
            log_det_init = x[:, :1] * 0.0
        else:
            log_det_init = torch.zeros(
                x.shape[0], 1,
                device=x.device,
                dtype=x.dtype,
                requires_grad=True
            )
            # Ensure x requires grad for adjoint method to work
            x = x.requires_grad_(True)
        state_init = torch.cat([x, log_det_init], dim=-1)

        # Ensure state_init requires grad when using adjoint_params
        # This is necessary for odeint_adjoint to build the computation graph
        if not state_init.requires_grad:
            state_init = state_init.requires_grad_(True)

        # Integrate augmented ODE
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
        # Ensure x requires grad for odeint_adjoint to work properly
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

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
        # Sample z ~ p(z)
        z = self.base_dist.sample((num_samples,))

        # Transform z -> x
        x, _ = self.forward(z, reverse=True)

        return x
