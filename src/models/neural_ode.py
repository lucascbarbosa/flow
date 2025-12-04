"""Neural ODE implementation using torchdiffeq."""
import torch
import torch.nn as nn
from torchdiffeq import odeint
from src.models.vector_field import VectorField
from typing import Literal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralODE(nn.Module):
    """Neural ODE: integrates dx/dt = f(x,t) from t=0 to t=1."""
    def __init__(
        self,
        vector_field: VectorField,
        solver: Literal['euler', 'rk4', 'dopri5'] = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ) -> None:
        """Initialize NeuralODE."""
        super().__init__()
        self.vf = vector_field
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(
        self,
        x: torch.Tensor,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """Integrate ODE from t=0 (x) to t=1 (z)."""
        # Create t_span with shape (n_steps, 1)
        t_span = torch.linspace(
            0., 1.,
            n_steps,
            device=device,
            dtype=torch.float64,
        )

        x_t = odeint(
            self.vf,
            x,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )
        return x_t

    def backward(
        self,
        x: torch.Tensor,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """Integrate ODE from t=1 to t=0."""
        t_span = torch.linspace(
            1., 0.,
            n_steps,
            device=device,
            dtype=torch.float64,
        )
        x_t = odeint(
            self.vf,
            x,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )
        return x_t

    def sample(
        self,
        n_samples: int = 100,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """Sample (z ~ N(0, I)) and integrate ODE from t=1 to t=0."""
        # Get dtype from model parameters
        z = torch.randn(
            n_samples,
            self.vf.features,
            device=device,
            dtype=torch.float64,
        )

        x_t = self.backward(z, n_steps)
        return x_t[-1]  # Return final state
