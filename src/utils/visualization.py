"""Visualization utilities."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from src.models.neural_ode import NeuralODE
from src.models.vector_field import VectorField
from src.models.cnf import CNF
from typing import Optional, Tuple, Union


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
    axes: Optional[Tuple[Axes, Axes]] = None
) -> Tuple[Axes, Axes]:
    """Plot trajectories x(t) from x(0) to x(1) for random dataset samples.

    Creates two plots:
    - Left: Data distribution (transparent) + sampled x(0) points with
      label colors
    - Right: Final state x(1) with label colors

    Args:
        model (NeuralODE): NeuralODE model.

        dataset (torch.Tensor | np.ndarray): Dataset with shape
            (n_samples, features) or (n_samples, features, ...).
            If tuple (data, labels), extracts data.

        n_samples (int): Number of random samples to plot.
            Default is 20.

        t_span (torch.Tensor): Time points for integration.

        n_points (int): Number of points in trajectory.

        axes (Tuple[Axes, Axes], optional): Two matplotlib axes for
            left and right plots. If None, creates new figure.

    Returns:
        Tuple[Axes, Axes]: Left and right matplotlib axes.
    """
    if axes is None:
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(16, 8)
        )
    else:
        ax_left, ax_right = axes

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

    # Get final state x(1)
    x_1 = x_t[-1]  # (n_plot, features)
    x_0_np = x_0.cpu().numpy()
    x_1_np = x_1.cpu().numpy()
    labels_np = labels.cpu().numpy()
    sampled_labels_np = sampled_labels.cpu().numpy()

    # Get unique labels and assign colors
    unique_labels = np.unique(labels_np)
    colors = plt.cm.get_cmap('tab10')(
        np.linspace(0, 1, len(unique_labels))
    )
    label_to_color = {
        label: colors[i] for i, label in enumerate(unique_labels)
    }

    # LEFT PLOT: Only sampled x(0) points with label colors
    for i, label in enumerate(unique_labels):
        mask = sampled_labels_np == label
        ax_left.scatter(
            x_0_np[mask, 0],
            x_0_np[mask, 1],
            alpha=0.5,
            s=10,
            color=label_to_color[label],
            label=f'Class {label}'
        )
    ax_left.set_xlabel(r'$x_1$')
    ax_left.set_ylabel(r'$x_2$')
    ax_left.set_title('Initial State x(0)')
    ax_left.grid(True, alpha=0.3)
    ax_left.axis('equal')
    ax_left.legend()

    # RIGHT PLOT: Final state x(1) with label colors
    for i, label in enumerate(unique_labels):
        mask = sampled_labels_np == label
        ax_right.scatter(
            x_1_np[mask, 0],
            x_1_np[mask, 1],
            alpha=0.5,
            s=10,
            color=label_to_color[label],
            label=f'Class {label}'
        )

    ax_right.set_xlabel(r'$x_1$')
    ax_right.set_ylabel(r'$x_2$')
    ax_right.set_title('Final State x(1)')
    ax_right.grid(True, alpha=0.3)
    ax_right.axis('equal')
    ax_right.legend()

    return ax_left, ax_right


def plot_vector_field(
    model: Union[VectorField, NeuralODE, CNF],
    xlim: Tuple[float, float] = (-2, 2),
    ylim: Tuple[float, float] = (-2, 2),
    n_grid: int = 20,
    t: float = 0.5,
    ax: Optional[Axes] = None
) -> Axes:
    """Plot vector field f(x, t) on a 2D grid.

    Args:
        model (Union[VectorField, NeuralODE, CNF]): VectorField model or
            model with .vf attribute.

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

    # Get vector field from model (handle both VectorField and models with .vf)
    if isinstance(model, VectorField):
        vf = model
    else:
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
    model: Union[NeuralODE, CNF],
    n_samples: int = 20,
    n_trajectory_points: int = 100,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    axes: Optional[Tuple[Axes, Axes]] = None
) -> Tuple[Axes, Axes]:
    """Plot trajectories starting at z ~ N(0, I) and transforming to x.

    Creates two plots:
    - Left: Initial z samples from N(0, I) (gray, no labels)
    - Right: Final transformed state x (gray, no labels)
      Note: True class labels are unknown when transforming from z ~ N(0, I)

    Works with both NeuralODE and CNF models.

    Args:
        model (Union[NeuralODE, CNF]): NeuralODE or CNF model.

        n_samples (int): Number of samples to plot trajectories for.
            Default is 20.

        n_trajectory_points (int): Number of points in each trajectory.
            Default is 100.

        xlim (Tuple[float, float]): Limits in x direction.

        ylim (Tuple[float, float]): Limits in y direction.

        axes (Tuple[Axes, Axes], optional): Two matplotlib axes for
            left and right plots. If None, creates new figure.

    Returns:
        Tuple[Axes, Axes]: Left and right matplotlib axes.
    """
    if axes is None:
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(16, 8)
        )
    else:
        ax_left, ax_right = axes

    # Get device and features from model
    device = next(model.vf.parameters()).device
    features = model.vf.features

    # Only support 2D visualization
    if features != 2:
        raise ValueError(
            f"plot_transformation only supports 2D models "
            f"(got {features}D)"
        )

    # Sample z ~ N(0, I)
    z = torch.randn(n_samples, features, device=device)

    model.eval()
    with torch.no_grad():
        t_span = torch.linspace(
            start=1.0,
            end=0.0,
            steps=n_trajectory_points,
            device=device
        )
        x_t, _ = model(z, t_span, reverse=True)

    # Get initial and final states
    z_np = z.cpu().numpy()

    # Handle different return formats:
    # - NeuralODE returns trajectory: [n_steps, n_samples, 2]
    # - CNF returns only final state: [n_samples, 2]
    if x_t.dim() == 3:
        # Full trajectory from NeuralODE: extract final state
        x_final_np = x_t[-1].cpu().numpy()  # [n_samples, 2]
    elif x_t.dim() == 2:
        # Already final state from CNF: use directly
        x_final_np = x_t.cpu().numpy()  # [n_samples, 2]
    else:
        raise ValueError(
            f"Unexpected x_t shape: {x_t.shape}. "
            f"Expected 2D [n_samples, 2] or 3D [n_steps, n_samples, 2]"
        )

    # LEFT PLOT: Initial z samples (gray, no labels)
    ax_left.scatter(
        z_np[:, 0],
        z_np[:, 1],
        color='gray',
        marker='o',
        s=50,
        alpha=0.6,
        linewidths=1
    )

    # Unit circle (reference)
    circle = Circle(
        (0, 0), 2, fill=False, linestyle='--',
        color='gray', alpha=0.5, label='|z|=2'
    )
    ax_left.add_patch(circle)

    ax_left.set_xlabel(r'$x_1$')
    ax_left.set_ylabel(r'$x_2$')
    ax_left.set_title('Sample points from normal z(0) ~ N(0,I)')
    ax_left.set_xlim(xlim)
    ax_left.set_ylim(ylim)
    ax_left.grid(True, alpha=0.3)
    ax_left.axis('equal')

    # RIGHT PLOT: Final transformed state (gray, no labels)
    # Note: We don't know true class labels when transforming from z ~ N(0, I)
    ax_right.scatter(
        x_final_np[:, 0],
        x_final_np[:, 1],
        color='gray',
        alpha=0.5,
        s=10
    )

    ax_right.set_xlabel(r'$x_1$')
    ax_right.set_ylabel(r'$x_2$')
    ax_right.set_title('Final state z(1)')
    ax_right.set_xlim(xlim)
    ax_right.set_ylim(ylim)
    ax_right.grid(True, alpha=0.3)
    ax_right.axis('equal')

    return ax_left, ax_right


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
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Convert to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Plot with or without labels
    if labels is not None:
        # Get unique labels and assign colors
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap(cmap)(
            np.linspace(0, 1, len(unique_labels))
        )

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
    else:
        # Plot without labels
        ax.scatter(
            data[:, 0],
            data[:, 1],
            alpha=0.5,
            s=10
        )
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    return ax


def plot_transformation_realnvp(
    flow,
    n_samples: int = 1000,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    axes: Optional[Tuple[Axes, Axes]] = None
) -> Tuple[Axes, Axes]:
    """Plot RealNVP transformation from base distribution to data distribution.

    Creates two plots:
    - Left: Initial z samples from N(0, I)
    - Right: Final transformed state x

    Args:
        flow: Zuko flow model (returns a Distribution when called).

        n_samples (int): Number of samples to plot. Default is 1000.

        xlim (Tuple[float, float]): Limits in x direction.

        ylim (Tuple[float, float]): Limits in y direction.

        axes (Tuple[Axes, Axes], optional): Two matplotlib axes for
            left and right plots. If None, creates new figure.

    Returns:
        Tuple[Axes, Axes]: Left and right matplotlib axes.
    """
    if axes is None:
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(16, 8)
        )
    else:
        ax_left, ax_right = axes

    # Get the distribution from the flow (unconditional)
    flow.eval()
    with torch.no_grad():
        dist = flow(None)

        # Sample z ~ N(0, I) from base distribution
        # In zuko, the base distribution is accessible via dist.base
        try:
            z = dist.base.sample((n_samples,))
        except AttributeError:
            # Fallback: sample manually from standard normal
            # Get dimension from transform
            from torch.distributions import MultivariateNormal
            dim = dist.transform.domain.event_shape[0]
            base = MultivariateNormal(
                torch.zeros(dim, device=next(flow.parameters()).device),
                torch.eye(dim, device=next(flow.parameters()).device)
            )
            z = base.sample((n_samples,))

        # Transform z -> x using the inverse transform
        # The flow's transform maps x -> z, so we use .inv() to map z -> x
        transform = dist.transform
        x = transform.inv(z)

    z_np = z.cpu().numpy()
    x_np = x.cpu().numpy()

    # LEFT PLOT: Initial z samples (gray, no labels)
    ax_left.scatter(
        z_np[:, 0],
        z_np[:, 1],
        color='gray',
        marker='o',
        s=50,
        alpha=0.6,
        linewidths=1
    )

    # Unit circle (reference)
    circle = Circle(
        (0, 0), 2, fill=False, linestyle='--',
        color='gray', alpha=0.5, label='|z|=2'
    )
    ax_left.add_patch(circle)

    ax_left.set_xlabel(r'$x_1$')
    ax_left.set_ylabel(r'$x_2$')
    ax_left.set_title('Sample points from normal z ~ N(0,I)')
    ax_left.set_xlim(xlim)
    ax_left.set_ylim(ylim)
    ax_left.grid(True, alpha=0.3)
    ax_left.axis('equal')

    # RIGHT PLOT: Final transformed state
    ax_right.scatter(
        x_np[:, 0],
        x_np[:, 1],
        color='gray',
        alpha=0.5,
        s=10
    )

    ax_right.set_xlabel(r'$x_1$')
    ax_right.set_ylabel(r'$x_2$')
    ax_right.set_title('Transformed samples x = T⁻¹(z)')
    ax_right.set_xlim(xlim)
    ax_right.set_ylim(ylim)
    ax_right.grid(True, alpha=0.3)
    ax_right.axis('equal')

    return ax_left, ax_right
