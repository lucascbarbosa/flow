"""FFJORD implementation."""
import torch
import torch.nn as nn
from src.models.vector_field import VectorField2D
from src.utils.trace import divergence_hutchinson
from torchdiffeq import odeint_adjoint
from torch.distributions import Distribution
from typing import Optional, Literal, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FFJORD(nn.Module):
    """Continuous Normalizing Flow with Hutchinson trace estimator."""
    def __init__(
        self,
        vector_field: VectorField2D,
        base_dist: Optional[Distribution] = None,
        num_samples: int = 1,
        distribution: Literal['rademacher', 'gaussian'] = 'rademacher'
    ) -> None:
        """Initialize FFJORD.

        Args:
            vector_field (VectorField2D): Vector field module f(x, t).

            base_dist (Distribution, optional): Base distribution.
                If None, uses N(0, I).

            num_samples (int): Number of samples for Hutchinson estimator.
                Default is 1.

            distribution (Literal['rademacher', 'gaussian']): Distribution for
                Hutchinson estimator. Default is 'rademacher'.
        """
        super().__init__()
        self.vf = vector_field
        self.num_samples = num_samples
        self.distribution = distribution

        if base_dist is None:
            # Prior: N(0, I)
            features = vector_field.features
            self.base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(features).to(device),
                torch.eye(features).to(device)
            )
        else:
            self.base_dist = base_dist.to(device)

    def _augmented_dynamics(
        self,
        t: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Augmented ODE: integrates [x, log_det] simultaneously.

        dx/dt = f(x, t)
        d(log_det)/dt = -trace(∂f/∂x) [estimated via Hutchinson estimator]

        Args:
            t (torch.Tensor): Scalar time.
            state (torch.Tensor): State tensor with shape (batch, features + 1)
                containing [x, log_det].

        Returns:
            torch.Tensor: Derivative with shape (batch, features + 1).
        """
        x = state[:, :-1]  # (batch, features)

        # Ensure x requires grad for divergence computation
        if not x.requires_grad:
            x = x.requires_grad_(True)

        # Compute vector field
        dx_dt = self.vf(t, x)  # (batch, features)

        # Compute trace of the Jacobian using Hutchinson estimator
        # Wrap in enable_grad to ensure gradient computation works properly
        with torch.enable_grad():
            trace = divergence_hutchinson(
                lambda x: self.vf(t, x),
                x,
                num_samples=self.num_samples,
                distribution=self.distribution
            )  # (batch,)

        # d(log_det)/dt = -trace (note the sign!)
        dlogdet_dt = -trace.unsqueeze(-1)  # (batch, 1)

        return torch.cat([dx_dt, dlogdet_dt], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        n_steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform x -> z (forward) by integrating from t=0 to t=1.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, features).
            n_steps (int): Number of time steps. Default is 10.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - z (torch.Tensor): Transformed tensor with shape
                    (batch, features).
                - log_det (torch.Tensor): Log determinant with shape
                    (batch,).
        """
        # x -> z: integrate from t=0 to t=1
        t_span = torch.linspace(
            0., 1.,
            n_steps,
            device=device,
            dtype=torch.float64,
        )

        # Ensure x is 2D: [batch, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Ensure x requires grad for proper gradient tracking through ODE
        if torch.is_grad_enabled():
            if not x.requires_grad:
                x = x.clone().requires_grad_(True)

        # Initial state: [x, log_det=0]
        # log_det_init should also require grad to maintain gradient flow
        # even though it starts at 0, it will accumulate gradients
        # during ODE integration
        log_det_init = torch.zeros(
            x.shape[0],
            1,
            device=device,
            dtype=torch.float64,
            requires_grad=False
        )
        state_init = torch.cat([x, log_det_init], dim=-1)

        # Integrate augmented ODE
        state_t = odeint_adjoint(
            self._augmented_dynamics,
            state_init,
            t_span,
            method='dopri5',
            rtol=1e-3,
            atol=1e-4,
            adjoint_params=tuple(self.vf.parameters())
        )

        # Final state
        state_final = state_t[-1]  # (batch, features + 1)
        z = state_final[:, :-1]  # (batch, features)
        log_det = state_final[:, -1]  # (batch,)

        return z, log_det

    def backward(
        self,
        x: torch.Tensor,
        n_steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform z -> x (backward) by integrating from t=1 to t=0.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, features).
            n_steps (int): Number of time steps. Default is 10.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - x_transformed (torch.Tensor): Transformed tensor with shape
                    (batch, features).
                - log_det (torch.Tensor): Log determinant with shape
                    (batch,).
        """
        # z -> x: integrate from t=1 to t=0
        t_span = torch.linspace(
            1., 0.,
            n_steps,
            device=device,
            dtype=torch.float64,
        )

        # Ensure x is 2D: [batch, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Ensure x requires grad for proper gradient tracking through ODE
        if torch.is_grad_enabled():
            if not x.requires_grad:
                x = x.clone().requires_grad_(True)

        # Initial state: [x, log_det=0]
        log_det_init = torch.zeros(
            x.shape[0],
            1,
            device=device,
            dtype=torch.float64,
            requires_grad=False
        )
        state_init = torch.cat([x, log_det_init], dim=-1)

        # Integrate augmented ODE
        state_t = odeint_adjoint(
            self._augmented_dynamics,
            state_init,
            t_span,
            method='dopri5',
            rtol=1e-3,
            atol=1e-4,
            adjoint_params=tuple(self.vf.parameters())
        )

        # Final state
        state_final = state_t[-1]  # (batch, features + 1)
        x_transformed = state_final[:, :-1]  # (batch, features)
        log_det = state_final[:, -1]  # (batch,)

        return x_transformed, log_det

    def log_prob(self, x: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """Calculate log p(x) using change of variables.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, features).
            n_steps (int): Number of time steps. Default is 10.

        Returns:
            torch.Tensor: Log probability with shape (batch,).
        """
        # Ensure x requires grad for odeint_adjoint to work properly
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

        # Transform x -> z
        z, log_det = self.forward(x, n_steps=n_steps)

        # log p(x) = log p(z) + log |det(∂z/∂x)|
        log_prob_z = self.base_dist.log_prob(z)
        log_prob_x = log_prob_z + log_det

        return log_prob_x

    def sample(self, num_samples: int, n_steps: int = 10) -> torch.Tensor:
        """Generate samples x ~ p(x) via z ~ p(z) -> x.

        Args:
            num_samples (int): Number of samples to generate.
            n_steps (int): Number of time steps. Default is 10.

        Returns:
            torch.Tensor: Samples with shape (num_samples, features).
        """
        # Sample z ~ p(z)
        z = self.base_dist.sample((num_samples,))

        # Transform z -> x using backward
        x, _ = self.backward(z, n_steps=n_steps)

        return x
