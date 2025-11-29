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
            X, y = make_moons(
                n_samples=n_samples, noise=noise, random_state=42
            )
        elif dataset_type == 'circles':
            X, y = make_circles(
                n_samples=n_samples,
                noise=noise,
                factor=0.5,
                random_state=42
            )
        elif dataset_type == 'spirals':
            X, y = self.make_spirals(
                n_samples=n_samples,
                noise=noise,
                random_state=42
            )
        self.data = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Returns:
            tuple: (data, label) where data has shape (features,) and
                label is a scalar tensor.
        """
        return self.data[idx], self.labels[idx]

    def make_spirals(
        self,
        n_samples: int = 1000,
        noise: float = 0.05,
        random_state: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Two intertwined spirals.

        Returns:
            tuple: (X, y) where X is the data and y are the labels.
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
        # Labels: first n samples are class 0, second n samples are class 1
        y_labels = np.hstack([
            np.zeros(n, dtype=np.int64),
            np.ones(n, dtype=np.int64)
        ])
        return X, y_labels


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

        self.data = torch.tensor(data_reduced, dtype=torch.float32)
        self.pca = pca

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample from the dataset."""
        return self.data[idx]


class MNISTFull(Dataset):
    """Full MNIST dataset with dequantization and logit preprocessing.

    Preprocessing steps:
    1. Dequantization: Add uniform noise to pixel values [0, 255]
    2. Normalize to [0, 1]
    3. Logit transform: logit(x) = log(x / (1 - x)) with alpha for stability
    """

    def __init__(self, train: bool = True, alpha: float = 1e-6) -> None:
        """Initialize full MNIST dataset.

        Args:
            train: If True, use training set.
            alpha: Small constant for logit stability. Default is 1e-6.
        """
        self.alpha = alpha

        # Load MNIST (raw pixels, no normalization)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to (784,)
        ])

        mnist = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

        # Convert to numpy and apply preprocessing
        data = []
        for i in range(len(mnist)):
            img = mnist[i][0].numpy()  # Already in [0, 1]
            data.append(img)
        data = np.array(data)

        # Store raw data (will apply dequantization and logit in __getitem__)
        self.raw_data = torch.tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample from the dataset with preprocessing.

        Applies:
        1. Dequantization: x = x + u where u ~ Uniform(0, 1/256)
        2. Clamp to [0, 1]
        3. Logit transform: logit(x) = log(x / (1 - x)) with alpha
        """
        x = self.raw_data[idx].clone()

        # Dequantization: add uniform noise
        u = torch.rand_like(x) / 256.0
        x = x + u

        # Clamp to [0, 1]
        x = torch.clamp(x, 0.0, 1.0)

        # Logit transform with alpha for numerical stability
        # logit(x) = log(x / (1 - x))
        # Use alpha to avoid log(0) or log(inf)
        x = x * (1 - 2 * self.alpha) + self.alpha
        x = torch.log(x / (1 - x))

        return x


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
