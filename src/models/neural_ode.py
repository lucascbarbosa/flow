"""Neural ODE implementation using torchdiffeq."""
import torch
import torch.nn as nn
from torchdiffeq import odeint
from src.models.vector_field import VectorField
from typing import Literal, Optional


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralODE(nn.Module):
    """Neural ODE: integrates dx/dt = f(x,t) from t=0 to t=1."""
    def __init__(
        self,
        vector_field: VectorField,
        solver: Literal['euler', 'rk4', 'dopri5'] = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4,
        num_classes: Optional[int] = None,
        hidden_dims: list[int] = [64, 32]
    ) -> None:
        """Initialize NeuralODE.

        Args:
            vector_field (VectorField): Vector field module f(x, t).

            solver (str, optional): ODE solver method. Default is 'dopri5'.

            rtol (float, optional): Relative tolerance. Default is 1e-3.

            atol (float, optional): Absolute tolerance. Default is 1e-4.

            num_classes (int, optional): Number of classes for
                classification. If provided, adds a classification head.
                Default is None.

            hidden_dims (list[int], optional): Hidden dimensions for MLP.
                Default is [64, 32].
        """
        super().__init__()
        self.vf = vector_field
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.num_classes = num_classes

        # Classification head (MLP)
        if num_classes is not None:
            features = vector_field.features
            dims = [features] + hidden_dims + [num_classes]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:  # No activation on last layer
                    layers.append(nn.ReLU())
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = None

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
