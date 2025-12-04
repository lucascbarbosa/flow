"""Vector Field architectures for Neural ODEs."""
import math
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VectorField(nn.Module):
    """Parametrizes dx/dt = f(x, t) using a neural network."""
    def __init__(
        self,
        features: int,
        hidden_dims: list[int] = [64, 64],
        time_embed_dim: int = 16
    ) -> None:
        """Initialize the VectorField.

        Args:
            features (int): Data dimension.

            hidden_dims (list, optional): Hidden dimensions.
                Default is [64, 64].

            time_embed_dim (int, optional): Time embedding dimension.
                Default is 16.
        """
        super().__init__()
        self.features = features
        self.time_embed_dim = time_embed_dim

        # Build MLP
        dims = [features + time_embed_dim] + hidden_dims + [features]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]).to(device))
            if i < len(dims) - 2:  # No activation on last layer
                layers.append(nn.SiLU().to(device))

        self.net = nn.Sequential(*layers)

        # Initialization: last layer with small weights (σ=0.01)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01).to(device)
        nn.init.zeros_(self.net[-1].bias).to(device)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding.

        Args:
            t (torch.Tensor): Time tensor with shape [batch] or [batch, 1]
                or scalar.

        Returns:
            embedded (torch.Tensor): Embedded time tensor with shape
                [batch, time_embed_dim].
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        elif t.dim() != 1:
            raise ValueError(
                f"Time tensor must have shape [batch, ], got {t.shape}"
            )

        elif t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]

        # t_emb[2i] = sin(t / 10000^(2i/d))
        # t_emb[2i+1] = cos(t / 10000^(2i/d))
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            - torch.arange(
                0, half,
                dtype=torch.float32,
                device=device
            )
            * math.log(10000.0) /
            (half - 1)
        )
        return torch.cat([
            torch.sin(t.unsqueeze(-1) * freqs),
            torch.cos(t.unsqueeze(-1) * freqs)
        ], dim=-1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Computes f(x, t) = dx/dt."""
        t = t.to(device)

        # Time embedding
        t_emb = self.time_embedding(t)

        if t_emb.shape[0] != x.shape[0]:
            if t_emb.shape[0] == 1:
                t_emb = t_emb.expand(x.shape[0], -1)
            else:
                raise ValueError(
                    f"Batch size mismatch: x {x.shape[0]}, "
                    f"t_emb {t_emb.shape[0]}"
                )

        # Concatenate [x, t_emb]
        x_t = torch.cat([x, t_emb], dim=-1)

        # Pass through the network
        dx_dt = self.net(x_t)

        return dx_dt
