"""Vector Field architectures for Neural ODEs."""
import math
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VectorField2D(nn.Module):
    """Parametrizes dx/dt = f(x, t) using a neural network."""
    def __init__(
        self,
        features: int,
        hidden_dims: list[int] = [64, 64],
        time_embed_dim: int = 16
    ) -> None:
        """Initialize the VectorField2D.

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
            layers.append(
                nn.Linear(
                    dims[i],
                    dims[i + 1],
                    dtype=torch.float64
                )
            )
            if i < len(dims) - 2:  # No activation on last layer
                layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

        # Initialization: last layer with small weights (σ=0.01)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding.

        Args:
            t (torch.Tensor): Time tensor with shape [batch] or [batch, 1]
                or scalar.

        Returns:
            embedded (torch.Tensor): Embedded time tensor with shape
                [batch, time_embed_dim].
        """
        # Handle scalar
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Handle 2D tensor with shape [batch, 1]
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]

        # Ensure 1D tensor
        elif t.dim() != 1:
            raise ValueError(
                f"Time tensor must have shape [batch, ], got {t.shape}"
            )

        # t_emb[2i] = sin(t / 10000^(2i/d))
        # t_emb[2i+1] = cos(t / 10000^(2i/d))
        half = self.time_embed_dim // 2
        # Standard positional encoding: 10000^(2i/d_model)
        # For i in [0, half-1], compute 10000^(-2i/d_model)
        i = torch.arange(0, half, dtype=torch.float64, device=t.device)
        freqs = torch.exp(-i * 2.0 * math.log(10000.0) / self.time_embed_dim)
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


class VectorFieldMNIST(nn.Module):
    """Parametrizes dx/dt = f(x, t) using CNN layers for image data (MNIST)."""
    def __init__(
        self,
        features: int = 784,
        image_size: int = 28,
        channels: list[int] = [32, 64, 128],
        time_embed_dim: int = 32,
        fc_hidden_dims: list[int] = [512, 512]
    ) -> None:
        """Initialize the VectorFieldMNIST.

        Args:
            features (int): Total number of features (28*28=784 for MNIST).
                Default is 784.

            image_size (int): Image size (assumes square images).
                Default is 28.

            channels (list[int]): Number of channels for each conv layer.
                Default is [32, 64, 128].

            time_embed_dim (int): Time embedding dimension.
                Default is 32.

            fc_hidden_dims (list[int]): Hidden dimensions for FC layers
                after CNN. Default is [512, 512].
        """
        super().__init__()
        self.features = features
        self.image_size = image_size
        self.time_embed_dim = time_embed_dim

        # Verify image_size matches features
        if image_size * image_size != features:
            raise ValueError(
                f"image_size^2 ({image_size**2}) must equal "
                f"features ({features})"
            )

        # Build CNN layers
        cnn_layers = []
        in_channels = 1  # MNIST is grayscale
        for out_channels in channels:
            cnn_layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    dtype=torch.float64
                ),
                nn.GroupNorm(8, out_channels, dtype=torch.float64),
                nn.SiLU()
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate flattened CNN output size
        # After conv layers, spatial size remains image_size x image_size
        cnn_output_size = channels[-1] * image_size * image_size

        # Build FC layers: [CNN_features + time_emb] -> fc_hidden -> features
        fc_dims = (
            [cnn_output_size + time_embed_dim] + fc_hidden_dims + [features]
        )
        fc_layers = []
        for i in range(len(fc_dims) - 1):
            fc_layers.append(
                nn.Linear(
                    fc_dims[i],
                    fc_dims[i + 1],
                    dtype=torch.float64
                )
            )
            if i < len(fc_dims) - 2:  # No activation on last layer
                fc_layers.append(nn.SiLU())

        self.fc = nn.Sequential(*fc_layers)

        # Initialization: last layer with small weights (σ=0.01)
        nn.init.normal_(self.fc[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc[-1].bias)

        # Initialize CNN layers with Xavier uniform
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding.

        Args:
            t (torch.Tensor): Time tensor with shape [batch] or [batch, 1]
                or scalar.

        Returns:
            embedded (torch.Tensor): Embedded time tensor with shape
                [batch, time_embed_dim].
        """
        # Handle scalar
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Handle 2D tensor with shape [batch, 1]
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]

        # Ensure 1D tensor
        elif t.dim() != 1:
            raise ValueError(
                f"Time tensor must have shape [batch, ], got {t.shape}"
            )

        # t_emb[2i] = sin(t / 10000^(2i/d))
        # t_emb[2i+1] = cos(t / 10000^(2i/d))
        half = self.time_embed_dim // 2
        i = torch.arange(0, half, dtype=torch.float64, device=t.device)
        freqs = torch.exp(-i * 2.0 * math.log(10000.0) / self.time_embed_dim)
        return torch.cat([
            torch.sin(t.unsqueeze(-1) * freqs),
            torch.cos(t.unsqueeze(-1) * freqs)
        ], dim=-1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Computes f(x, t) = dx/dt.

        Args:
            t (torch.Tensor): Time tensor with shape [batch] or scalar.
            x (torch.Tensor): Input tensor with shape [batch, features]
                (flattened images).

        Returns:
            dx_dt (torch.Tensor): Output tensor with shape [batch, features].
        """
        t = t.to(device)

        batch_size = x.shape[0]
        input_features = x.shape[1]

        # Validate input size matches expected features
        if input_features != self.features:
            raise ValueError(
                f"Input features ({input_features}) does not match "
                f"expected features ({self.features}). "
                f"VectorFieldMNIST expects {self.features} features "
                f"({self.image_size}x{self.image_size} images). "
                f"For reduced-dimension data, use VectorField2D instead."
            )

        # Reshape flattened input to image format
        # (batch, features) -> (batch, 1, image_size, image_size)
        x_img = x.view(batch_size, 1, self.image_size, self.image_size)

        # Pass through CNN
        # (batch, channels[-1], 28, 28)
        cnn_features = self.cnn(x_img)
        # (batch, channels[-1]*28*28)
        cnn_features_flat = cnn_features.view(batch_size, -1)

        # Time embedding
        t_emb = self.time_embedding(t)

        if t_emb.shape[0] != batch_size:
            if t_emb.shape[0] == 1:
                t_emb = t_emb.expand(batch_size, -1)
            else:
                raise ValueError(
                    f"Batch size mismatch: x {batch_size}, "
                    f"t_emb {t_emb.shape[0]}"
                )

        # Concatenate CNN features with time embedding
        x_t = torch.cat([cnn_features_flat, t_emb], dim=-1)

        # Pass through FC layers
        dx_dt = self.fc(x_t)  # (batch, features)

        return dx_dt
