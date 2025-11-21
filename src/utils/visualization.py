"""Visualization utilities."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from src.models.neural_ode import NeuralODE
from src.models.vector_field import VectorField
from typing import Optional, Tuple


def plot_trajectories(
    model: NeuralODE,
    z: torch.Tensor,
    t_span: Optional[torch.Tensor] = None,
    n_points: int = 100,
    ax: Optional[Axes] = None
) -> Axes:
    """Plot trajectories x(t) for different initial conditions.

    Args:
        model (NeuralODE): NeuralODE model.

        z (torch.Tensor): Initial conditions with shape (batch, 2).

        t_span (torch.Tensor): Time points for integration.

        n_points (int): Number of points in trajectory.

        ax (Axes): Matplotlib axis.

    Returns:
        Axes: Matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    if t_span is None:
        t_span = torch.linspace(0, 1, n_points)

    # Integrate ODE
    model.eval()
    with torch.no_grad():
        x_t = model(z, t_span)

    # Plot trajectories with legends for start, end, trajectories
    x_t_np = x_t.cpu().numpy()

    for i in range(z.shape[0]):
        # Only set label for the first trajectory for legend clarity
        traj_label = "Trajectory" if i == 0 else None
        start_label = "Start Point" if i == 0 else None
        end_label = "End Point" if i == 0 else None

        ax.plot(
            x_t_np[:, i, 0],
            x_t_np[:, i, 1],
            alpha=0.6,
            linewidth=1.5,
            color='blue',
            label=traj_label,
        )
        # Mark start
        ax.scatter(
            x_t_np[0, i, 0],
            x_t_np[0, i, 1],
            color='green',
            marker='o',
            s=50,
            zorder=5,
            label=start_label
        )
        # Mark end
        ax.scatter(
            x_t_np[-1, i, 0],
            x_t_np[-1, i, 1],
            color='red',
            marker='s',
            s=50,
            zorder=5,
            label=end_label
        )

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('ODE Trajectories')
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

    # Convert t to tensor and extract scalar value for title
    if isinstance(t, torch.Tensor):
        if t.dim() == 0:
            t_value = t.item()
            t_tensor = t.to(torch.float32)
        elif t.dim() == 1:
            t_value = t[-1].item() if len(t) > 0 else t[0].item()
            t_tensor = (
                t[-1].to(torch.float32) if len(t) > 0
                else t[0].to(torch.float32)
            )
        else:
            t_value = t[0, -1].item()
            t_tensor = t[0, -1].to(torch.float32)
    else:
        t_value = float(t)
        t_tensor = torch.tensor(t_value, dtype=torch.float32)

    # Create grid
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)

    # Convert to tensor
    grid_points = torch.tensor(
        np.stack([X.ravel(), Y.ravel()], axis=1),
        dtype=torch.float32
    )

    # Calculate vector field
    with torch.no_grad():
        if hasattr(model, 'vf'):
            vf = model.vf
        else:
            vf = model

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

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'Vector Field at t={t_value:.2f}')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    return ax


def plot_transformation(
    model: NeuralODE,
    n_samples: int = 1000,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    ax: Optional[Axes] = None
) -> Axes:
    """Plot transformation z ~ N(0, I) -> x = φ(z, 1).

    Args:
        model (NeuralODE): NeuralODE model.

        n_samples (int): Number of samples.

        xlim (Tuple[float, float]): Limits in x direction.

        ylim (Tuple[float, float]): Limits in y direction.

        ax (Axes): Matplotlib axis.

    Returns:
        Axes: Matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Sample z ~ N(0, I)
    z = torch.randn(n_samples, 2)

    # Transform z -> x
    with torch.no_grad():
        if hasattr(model, 'forward') and hasattr(model, 'base_dist'):
            # CNF: use reverse=True
            x, _ = model.forward(z, reverse=True)
        else:
            # NeuralODE: integrate from t=0 to t=1
            x_t = model(z)
            x = x_t[-1]

    # Plot
    z_np = z.cpu().numpy()
    x_np = x.cpu().numpy()

    ax.scatter(
        z_np[:, 0], z_np[:, 1], alpha=0.5, s=10,
        label='z ~ N(0,I)', color='blue'
    )
    ax.scatter(
        x_np[:, 0], x_np[:, 1], alpha=0.5, s=10,
        label='x = φ(z,1)', color='red'
    )

    # Unit circle (reference)
    circle = Circle(
        (0, 0), 2, fill=False, linestyle='--',
        color='gray', alpha=0.5, label='|z|=2'
    )
    ax.add_patch(circle)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Transformation z -> x')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    return ax


def plot_data_distribution(
    data: torch.Tensor | np.ndarray,
    ax: Optional[Axes] = None,
    title: str = 'Data Distribution'
) -> Axes:
    """Plot data distribution.

    Args:
        data (torch.Tensor | np.ndarray): Data tensor or array with shape
            (n_samples, 2).
        ax (Axes): Matplotlib axis.

        title (str): Plot title.

    Returns:
        Axes: Matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    return ax
