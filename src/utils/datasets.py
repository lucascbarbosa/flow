"""Data loading utilities."""
import torch
import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from typing import Literal


class Synthetic2D(Dataset):
    """Synthetic 2D dataset (moons, circles, etc.)."""

    def __init__(
        self,
        n_samples: int = 5000,
        noise: float = 0.05,
        dataset_type: Literal['moons', 'circles', 'spirals'] = 'moons'
    ) -> None:
        """Initialize synthetic 2D dataset.

        Args:
            n_samples (int): Number of samples. Default is 5000.

            noise (float): Noise level. Default is 0.05.

            dataset_type (Literal['moons', 'circles', 'spirals']): Dataset
                type. Default is 'moons'.

        """
        if dataset_type == 'moons':
            X, _ = make_moons(
                n_samples=n_samples, noise=noise, random_state=42
            )
        elif dataset_type == 'circles':
            X, _ = make_circles(
                n_samples=n_samples,
                noise=noise,
                factor=0.5,
                random_state=42
            )
        elif dataset_type == 'spirals':
            X = self.make_spirals(
                n_samples=n_samples,
                noise=noise,
                random_state=42
            )
        self.data = torch.tensor(X, dtype=torch.float64)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample from the dataset.

        Returns:
            torch.Tensor: Data with shape (features,).
        """
        return self.data[idx]

    def make_spirals(
        self,
        n_samples: int = 1000,
        noise: float = 0.05,
        random_state: int | None = None
    ) -> np.ndarray:
        """Two intertwined spirals.

        Returns:
            np.ndarray: Data with shape (n_samples, 2).
        """
        if random_state is not None:
            np.random.seed(random_state)
        n = n_samples // 2
        theta = np.sqrt(np.random.rand(n)) * 2 * np.pi

        r = theta / (2 * np.pi)
        x = r * np.cos(theta) + noise * np.random.randn(n)
        y = r * np.sin(theta) + noise * np.random.randn(n)

        # Second spiral (rotated)
        x2 = -r * np.cos(theta) + noise * np.random.randn(n)
        y2 = -r * np.sin(theta) + noise * np.random.randn(n)
        X = np.vstack([np.column_stack([x, y]), np.column_stack([x2, y2])])
        return X


class MNISTReduced(Dataset):
    """MNIST reduced using PCA (top 100 pixels)."""

    def __init__(self, train: bool = True, n_components: int = 100) -> None:
        """Initialize reduced MNIST dataset.

        Args:
            train: If True, use training set.
            n_components: Number of PCA components.
        """
        # Load MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten
        ])

        mnist = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

        # Convert to numpy
        data = []
        for i in range(len(mnist)):
            data.append(mnist[i][0].numpy())
        data = np.array(data)

        # Apply PCA
        pca = PCA(n_components=n_components)
        data_reduced = pca.fit_transform(data)

        self.data = torch.tensor(data_reduced, dtype=torch.float64)
        self.pca = pca

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample from the dataset."""
        return self.data[idx]


class MNISTComplete(Dataset):
    """MNIST complete dataset (full 784 pixels, no PCA reduction)."""

    def __init__(self, train: bool = True) -> None:
        """Initialize complete MNIST dataset.

        Args:
            train: If True, use training set.
        """
        # Load MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten
        ])

        mnist = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

        # Convert to tensor
        data = []
        for i in range(len(mnist)):
            data.append(mnist[i][0])
        data = torch.stack(data)

        # Apply preprocessing
        self.data = self.preprocess_mnist(data)

    @staticmethod
    def preprocess_mnist(x: torch.Tensor) -> torch.Tensor:
        """Preprocess MNIST data with dequantization and logit transform.

        Args:
            x: Input tensor with values in [0, 1] (from ToTensor).

        Returns:
            Preprocessed tensor with logit-transformed values.
        """
        # Scale back to [0, 255] range for dequantization
        x = x * 255.0

        # Dequantization: x ∈ {0,...,255} → x ∈ (0, 256)
        x = x + torch.rand_like(x)
        x = x / 256.0  # → (0, 1)

        # Logit transform para evitar boundary issues
        alpha = 0.05
        x = alpha + (1 - 2 * alpha) * x
        x = torch.logit(x)

        return x

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample from the dataset."""
        return self.data[idx]


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = True
) -> DataLoader:
    """Create DataLoader.

    Args:
        dataset: Dataset to load.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.

    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
