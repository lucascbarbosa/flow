"""Visualization utilities."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from src.models.neural_ode import NeuralODE
from src.models.vector_field import VectorField
from typing import Optional, Tuple


def _darken_color(color: np.ndarray, factor: float = 0.7) -> np.ndarray:
    """Darken an RGBA color by multiplying RGB values by a factor.

    Args:
        color: RGBA color array with values in [0, 1].
        factor: Darkening factor (0-1). Default is 0.7.

    Returns:
        Darkened RGBA color array.
    """
    darkened = color.copy()
    darkened[:3] *= factor  # Only darken RGB, keep alpha
    return darkened


def plot_trajectories(
    model: NeuralODE,
    dataset: torch.Tensor | np.ndarray,
    n_samples: int = 20,
    t_span: Optional[torch.Tensor] = None,
    n_points: int = 100,
    ax: Optional[Axes] = None
) -> Axes:
    """Plot trajectories x(t) from x(0) to x(1) for random dataset samples.

    Shows trajectories starting from random samples in the dataset,
    integrating from t=0 to t=1.

    Args:
        model (NeuralODE): NeuralODE model.

        dataset (torch.Tensor | np.ndarray): Dataset with shape
            (n_samples, features) or (n_samples, features, ...).
            If tuple (data, labels), extracts data.

        n_samples (int): Number of random samples to plot.
            Default is 20.

        t_span (torch.Tensor): Time points for integration.

        n_points (int): Number of points in trajectory.

        ax (Axes): Matplotlib axis.

    Returns:
        Axes: Matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    data = dataset.data
    labels = dataset.labels

    # Sample random subset
    n_total = data.shape[0]
    n_plot = min(n_samples, n_total)
    indices = torch.randperm(n_total)[:n_plot]
    x_0 = data[indices].to(model.vf.net[0].weight.device)
    sampled_labels = labels[indices]

    if t_span is None:
        t_span = torch.linspace(0, 1, n_points).to(x_0.device)

    # Integrate ODE from x(0) to x(1)
    model.eval()
    with torch.no_grad():
        x_t, _ = model(x_0, t_span)

    # Plot trajectories with color coding by label
    x_t_np = x_t.cpu().numpy()
    sampled_labels_np = sampled_labels.cpu().numpy()

    # Get unique labels and assign colors
    unique_labels = np.unique(sampled_labels_np)
    colors = plt.cm.get_cmap('tab10')(
        np.linspace(0, 1, len(unique_labels))
    )
    # Darken colors for trajectories, x(0), and x(1)
    label_to_color = {
        label: _darken_color(colors[i])
        for i, label in enumerate(unique_labels)
    }

    # Track which labels we've already added to legend for each class
    start_plotted = {label: False for label in unique_labels}
    end_plotted = {label: False for label in unique_labels}

    for i in range(x_0.shape[0]):
        label = sampled_labels_np[i]
        color = label_to_color[label]

        # Only set label for the first trajectory of each class
        start_label = (
            fr"$x_{label}(0)$" if not start_plotted[label] else None
        )
        end_label = (
            fr"$x_{label}(1)$" if not end_plotted[label] else None
        )

        ax.plot(
            x_t_np[:, i, 0],
            x_t_np[:, i, 1],
            alpha=0.6,
            linewidth=1.5,
            color=color,
        )

        # Mark start x(0) with same color
        ax.scatter(
            x_t_np[0, i, 0],
            x_t_np[0, i, 1],
            color=color,
            marker='o',
            s=50,
            zorder=5,
            label=start_label
        )
        if start_label:
            start_plotted[label] = True

        # Mark end x(1) with same color
        ax.scatter(
            x_t_np[-1, i, 0],
            x_t_np[-1, i, 1],
            color=color,
            marker='s',
            s=50,
            zorder=5,
            label=end_label
        )
        if end_label:
            end_plotted[label] = True

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title('Trajectories from x(0) to x(1)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    return ax


def plot_vector_field(
    model: VectorField,
    xlim: Tuple[float, float] = (-2, 2),
    ylim: Tuple[float, float] = (-2, 2),
    n_grid: int = 20,
    t: float = 0.5,
    ax: Optional[Axes] = None
) -> Axes:
    """Plot vector field f(x, t) on a 2D grid.

    Args:
        model (VectorField): VectorField model.

        xlim (Tuple[float, float]): Limits in x direction.

        ylim (Tuple[float, float]): Limits in y direction.

        n_grid (int): Number of grid points.

        t (float): Time to evaluate.

        ax (Axes): Matplotlib axis.

    Returns:
        Axes: Matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Get device from model
    vf = model.vf
    device = next(vf.parameters()).device

    # Convert t to tensor and extract scalar value for title
    t_value = float(t)
    t_tensor = torch.tensor(
        t_value, dtype=torch.float32, device=device
    )

    # Create grid
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)

    # Convert to tensor
    grid_points = torch.tensor(
        np.stack([X.ravel(), Y.ravel()], axis=1),
        dtype=torch.float32,
        device=device
    )

    # Calculate vector field
    with torch.no_grad():
        dx_dt = vf(t_tensor, grid_points).cpu().numpy()

    # Reshape
    U = dx_dt[:, 0].reshape(X.shape)
    V = dx_dt[:, 1].reshape(Y.shape)

    # Normalize for visualization
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 1e-8)
    V_norm = V / (magnitude + 1e-8)

    # Plot
    ax.quiver(
        X, Y, U_norm, V_norm, magnitude,
        cmap='viridis', scale=20, width=0.005, alpha=0.7
    )

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(f'Vector Field at t={t_value:.2f}')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    return ax


def plot_transformation(
    model: NeuralODE,
    n_samples: int = 20,
    n_trajectory_points: int = 100,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    ax: Optional[Axes] = None
) -> Axes:
    """Plot trajectories starting at z ~ N(0, I) and transforming to x.

    Shows full trajectories from z(0) ~ N(0, I) to x(1) = φ(z, 1).

    Args:
        model (NeuralODE): NeuralODE model.

        n_samples (int): Number of samples to plot trajectories for.
            Default is 20.

        n_trajectory_points (int): Number of points in each trajectory.
            Default is 100.

        xlim (Tuple[float, float]): Limits in x direction.

        ylim (Tuple[float, float]): Limits in y direction.

        ax (Axes): Matplotlib axis.

    Returns:
        Axes: Matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Get device from model
    device = next(model.vf.parameters()).device

    # Sample z ~ N(0, I)
    z = torch.randn(n_samples, 2, device=device)

    # Create time span
    t_span = torch.linspace(0, 1, n_trajectory_points, device=device)

    # Transform z -> x (integrate ODE from t=0 to t=1)
    model.eval()
    with torch.no_grad():
        x_t, _ = model(z, t_span)

    # Plot trajectories
    x_t_np = x_t.cpu().numpy()

    for i in range(n_samples):
        # Only set label for the first trajectory
        traj_label = "Trajectory" if i == 0 else None
        start_label = "z(0) ~ N(0,I)" if i == 0 else None
        end_label = "x(1)" if i == 0 else None

        # Plot trajectory path
        ax.plot(
            x_t_np[:, i, 0],
            x_t_np[:, i, 1],
            alpha=0.4,
            linewidth=1.5,
            color='blue',
            label=traj_label,
        )
        # Mark start z(0)
        ax.scatter(
            x_t_np[0, i, 0],
            x_t_np[0, i, 1],
            color='green',
            marker='o',
            s=50,
            zorder=5,
            label=start_label
        )
        # Mark end x(1)
        ax.scatter(
            x_t_np[-1, i, 0],
            x_t_np[-1, i, 1],
            color='red',
            marker='s',
            s=50,
            zorder=5,
            label=end_label
        )

    # Unit circle (reference)
    circle = Circle(
        (0, 0), 2, fill=False, linestyle='--',
        color='gray', alpha=0.5, label='|z|=2'
    )
    ax.add_patch(circle)

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title('Trajectories from z(0) ~ N(0,I) to x(1)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    return ax


def plot_data_distribution(
    data: torch.Tensor | np.ndarray,
    labels: Optional[torch.Tensor | np.ndarray] = None,
    ax: Optional[Axes] = None,
    title: str = 'Data Distribution',
    cmap: str = 'tab10'
) -> Axes:
    """Plot data distribution.

    Args:
        data (torch.Tensor | np.ndarray): Data tensor or array with shape
            (n_samples, 2).

        labels (torch.Tensor | np.ndarray, optional): Labels tensor or array
            with shape (n_samples,). If provided, points are colored by label.
            Default is None.

        ax (Axes): Matplotlib axis.

        title (str): Plot title.

        cmap (str): Colormap name for coloring by labels.
            Default is 'tab10'.

    Returns:
        Axes: Matplotlib axis.
    """
    data = data.cpu().numpy()
    labels = labels.cpu().numpy()

    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))

    # Plot each label with different color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            data[mask, 0],
            data[mask, 1],
            alpha=0.5,
            s=10,
            color=colors[i],
            label=f'Class {label}'
        )
    ax.legend()
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    return ax
