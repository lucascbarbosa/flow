"""Neural ODE implementation using torchdiffeq."""
import torch
import torch.nn as nn
from torchdiffeq import odeint
from src.models.vector_field import VectorField
from typing import Literal


class NeuralODE(nn.Module):
    """Neural ODE: integrates dx/dt = f(x,t) from t=0 to t=1."""
    def __init__(
        self,
        vector_field: VectorField,
        solver: Literal['euler', 'rk4', 'dopri5'] = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4
    ) -> None:
        """Initialize NeuralODE.

        Args:
            vector_field (VectorField): Vector field module f(x, t).

            solver (str, optional): ODE solver method. Default is 'dopri5'.

            rtol (float, optional): Relative tolerance. Default is 1e-3.

            atol (float, optional): Absolute tolerance. Default is 1e-4.
        """
        super().__init__()
        self.vf = vector_field
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(
        self,
        z: torch.Tensor,
        t_span: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Integrate ODE from t=0 to t=1.

        Args:
            z (torch.Tensor): Initial state with shape (batch, features).
            t_span (torch.Tensor, optional): Time points to evaluate.
                Default is [0, 1].

        Returns:
            x_t (torch.Tensor): Trajectory with shape
                (len(t_span), batch, features).
        """
        if t_span is None:
            t_span = torch.linspace(0, 1, 100).to(z.device)

        # Use odeint from torchdiffeq
        # The vector field must accept (t, x) where t is scalar
        x_t = odeint(
            self.vf,
            z,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )

        return x_t
