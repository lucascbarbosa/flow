"""Vector Field architectures for Neural ODEs."""
import math
import torch
import torch.nn as nn


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
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on last layer
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

        # Initialization: last layer with small weights (σ=0.01)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding.

        Args:
            t (torch.Tensor): Time tensor with shape [batch, 1] or scalar.

        Returns:
            embedded (torch.Tensor): Embedded time tensor with shape
                [batch, time_embed_dim].
        """
        # Expand to batch if necessary
        if t.dim() == 0:
            t = t.unsqueeze(0)

        batch_size = t.shape[0]
        device = t.device

        # t_emb[2i] = sin(t / 10000^(2i/d))
        # t_emb[2i+1] = cos(t / 10000^(2i/d))
        div_term = torch.exp(
            torch.arange(
                0, self.time_embed_dim, 2,
                dtype=torch.float32, device=device
            )
            * (-math.log(10000.0) / self.time_embed_dim)
        )

        t_emb = torch.zeros(batch_size, self.time_embed_dim, device=device)
        t_emb[:, 0::2] = torch.sin(t.unsqueeze(-1) * div_term)
        t_emb[:, 1::2] = torch.cos(t.unsqueeze(-1) * div_term)

        return t_emb

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Computes f(x, t) = dx/dt.

        Args:
            t (torch.Tensor): Time tensor with shape [batch, 1] or scalar.
            x (torch.Tensor): Input tensor with shape [batch, features].

        Returns:
            dx_dt (torch.Tensor): Output tensor with shape [batch, features].
        """
        # Handle different t shapes from odeint
        if t.dim() == 0:
            # Scalar: expand to batch
            t = t.expand(x.shape[0])
        elif t.dim() == 1 and t.shape[0] == 1:
            # 1D tensor with single element: expand to batch
            t = t[0].expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            # If t has different batch size, assume it is the same for all batch
            t = t[0].expand(x.shape[0])

        # Time embedding
        t_emb = self.time_embedding(t)

        # Concatenate [x, t_emb]
        x_t = torch.cat([x, t_emb], dim=-1)

        # Pass through the network
        dx_dt = self.net(x_t)

        return dx_dt


class ResNetVF(VectorField):
    """Vector Field with skip connections (inspired by ResNet)."""
    def __init__(
        self,
        features: int,
        hidden_dims: list[int] = [64, 64],
        time_embed_dim: int = 16
    ) -> None:
        """Initialize ResNetVF.

        Args:
            features (int): Data dimension.

            hidden_dims (list, optional): Hidden dimensions.
                Default is [64, 64].

            time_embed_dim (int, optional): Time embedding dimension.
                Default is 16.
        """
        super().__init__(features, hidden_dims, time_embed_dim)

        # Override network with residual blocks
        self.input_dim = features + time_embed_dim
        self.hidden_dims = hidden_dims

        # First layer
        self.first_layer = nn.Linear(self.input_dim, hidden_dims[0])

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                in_dim = hidden_dims[0]
            else:
                in_dim = hidden_dims[i - 1]
            out_dim = hidden_dims[i]

            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Tanh(),
                nn.Linear(out_dim, out_dim)
            )
            self.res_blocks.append(block)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], features)

        # Initialization
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward with skip connections."""
        # Time embedding
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            t = t[0].expand(x.shape[0])

        t_emb = self.time_embedding(t)
        x_t = torch.cat([x, t_emb], dim=-1)

        # First layer
        h = torch.tanh(self.first_layer(x_t))

        # Residual blocks
        for block in self.res_blocks:
            h_new = torch.tanh(block[0](h))
            h_new = block[2](h_new)
            h = h + h_new  # Skip connection
            h = torch.tanh(h)

        # Output
        dx_dt = self.output_layer(h)

        return dx_dt
