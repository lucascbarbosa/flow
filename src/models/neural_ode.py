"""Neural ODE implementation using torchdiffeq."""
import torch
import torch.nn as nn
from torchdiffeq import odeint
from src.models.vector_field import VectorField
from typing import Literal, Optional


class NeuralODE(nn.Module):
    """Neural ODE: integrates dx/dt = f(x,t) from t=0 to t=1."""
    def __init__(
        self,
        vector_field: VectorField,
        solver: Literal['euler', 'rk4', 'dopri5'] = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4,
        num_classes: Optional[int] = None,
        classifier_hidden_dims: list[int] = [64, 32]
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

            classifier_hidden_dims (list[int], optional): Hidden dimensions
                for classification MLP. Default is [64, 32].
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
            dims = [features] + classifier_hidden_dims + [num_classes]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:  # No activation on last layer
                    layers.append(nn.ReLU())
            self.classifier = nn.Sequential(*layers)
        else:
            self.classifier = None

    def forward(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor | None = None,
        reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate ODE from t=0 to t=1 (forward) or t=1 to t=0 (reverse).

        Args:
            x (torch.Tensor): Initial state with shape (batch, features).
            t_span (torch.Tensor, optional): Time points to evaluate.
                Default is [0, 1] for forward or [1, 0] for reverse.
            reverse (bool): If True, integrates from t=1 to t=0 (reverse).
                Default is False (forward from t=0 to t=1).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - x_t (torch.Tensor): Trajectory with shape
                    (len(t_span), batch, features).
                - logits (torch.Tensor): Classification logits with shape
                    (batch, num_classes).
        """
        if t_span is None:
            if reverse:
                # z -> x: integrate from t=1 to t=0
                t_span = torch.tensor([1., 0.], device=x.device, dtype=x.dtype)
            else:
                # x -> z: integrate from t=0 to t=1
                t_span = torch.tensor([0., 1.], device=x.device, dtype=x.dtype)
        else:
            t_span = t_span.to(x.device)

        # Use odeint from torchdiffeq
        # The vector field must accept (t, x) where t is scalar
        x_t = odeint(
            self.vf,
            x,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )

        x_final = x_t[-1]  # Final state: (batch, features)

        # Classify final state if classifier exists
        if self.classifier is not None:
            # (batch, num_classes)
            logits = self.classifier(x_final)
        else:
            # Dummy logits if no classifier
            # (shouldn't happen in classification)
            logits = torch.zeros(
                x_final.shape[0], 1, device=x_final.device
            )

        return x_t, logits
