"""Visualization utilities."""
import imageio
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from src.models.neural_ode import NeuralODE
from src.models.vector_field import VectorField2D
from src.models.cnf import CNF
from src.models.ffjord import FFJORD
from typing import Optional, Tuple, Union, List
from zuko.flows import RealNVP


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Synthetic2DViz:
    """Visualization class for 2D synthetic datasets."""
    @classmethod
    def plot_trajectories(
        cls,
        model: Union[NeuralODE, CNF, RealNVP],
        dataset,
        n_samples: int = 20,
        n_steps: int = 100,
        axes: Optional[Tuple[Axes, Axes]] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Axes, Axes]:
        """Plot trajectories x(t) from x(0) to x(1) for random dataset samples.

        Creates two plots:
        - Left: Data distribution (transparent) + sampled x(0) points with
          label colors
        - Right: Final state x(1) with label colors

        Args:
            model (Union[NeuralODE, CNF, RealNVP]): Model. NeuralODE, CNF or
                RealNVP for trajectory integration.

            dataset: Dataset with .data and .labels attributes.

            n_samples (int): Number of random samples to plot.
                Default is 20.

            n_steps (int): Number of time steps. Default is 100.

            axes (Tuple[Axes, Axes], optional): Two matplotlib axes for
                left and right plots. If None, creates new figure.

            save_path (str, optional): Path to save the figure. If provided,
                saves the figure to the specified path. Default is None.

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

        # Sample random subset
        n_total = data.shape[0]
        n_plot = min(n_samples, n_total)
        indices = torch.randperm(n_total)[:n_plot]

        x_0 = data[indices].to(device)

        # Integrate ODE from x(0) to x(1)
        model.eval()
        x_t = model(x_0, n_steps)

        x_1 = x_t[-1]
        x_0_np = x_0.detach().cpu().numpy()
        x_1_np = x_1.detach().cpu().numpy()

        # LEFT PLOT: Only sampled x(0) points with label colors
        ax_left.scatter(
            x_0_np[:, 0],
            x_0_np[:, 1],
            alpha=0.5,
            s=10,
        )
        ax_left.set_xlabel(r'$x_1$')
        ax_left.set_ylabel(r'$x_2$')
        ax_left.set_title('Initial State x(0)')
        ax_left.grid(True, alpha=0.3)
        ax_left.axis('equal')
        ax_left.legend()

        # RIGHT PLOT: Final state x(1) with label colors
        ax_right.scatter(
            x_1_np[:, 0],
            x_1_np[:, 1],
            alpha=0.5,
            s=10,
        )
        # Add unit circle for N(0, I) reference
        circle_right = Circle(
            (0, 0),
            1,
            fill=False,
            linestyle='--',
            color='gray',
        )
        ax_right.add_patch(circle_right)
        ax_right.set_xlabel(r'$x_1$')
        ax_right.set_ylabel(r'$x_2$')
        ax_right.set_title('Final State x(1)')
        ax_right.grid(True, alpha=0.3)
        ax_right.axis('equal')
        ax_right.legend()

        # Save figure if path is provided
        if save_path is not None:
            fig = ax_left.figure
            # Ensure directory exists
            dirname = (
                os.path.dirname(save_path)
                if os.path.dirname(save_path) else '.'
            )
            os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return ax_left, ax_right

    @classmethod
    def plot_vector_field(
        cls,
        model: Union[
            NeuralODE,
            CNF,
            FFJORD,
            RealNVP,
            VectorField2D,
            List[Union[NeuralODE, CNF, FFJORD, RealNVP, VectorField2D]]],
        xlim: Tuple[float, float] = (-2, 2),
        ylim: Tuple[float, float] = (-2, 2),
        n_grid: int = 20,
        t: float = 0.5,
        direction: str = 'forward',
        ax: Optional[Axes] = None,
        save_path: Optional[Union[str, List[str]]] = None
    ) -> Union[Axes, List[Axes]]:
        """Plot vector field f(x, t) on a 2D grid.

        If a list of models is provided, creates a plot for each model.
        If save_path is a list, saves each plot to the corresponding path.

        Args:
            model: Single model or list of models.
                VectorField2D, NeuralODE, CNF, or FFJORD for ODE-based models.
                RealNVP for flow-based models.

            xlim (Tuple[float, float]): Limits in x direction.

            ylim (Tuple[float, float]): Limits in y direction.

            n_grid (int): Number of grid points.

            t (float): Time to evaluate (for ODE models).

            direction (str): For RealNVP, 'forward' or 'inverse'.
                Default is 'forward'.

            ax (Axes): Matplotlib axis (only used for single model).

            save_path: Single path or list of paths for saving figures.

        Returns:
            Single Axes or list of Axes.
        """
        # Handle list of models
        if isinstance(model, list):
            if save_path is None:
                save_paths = [None] * len(model)
            elif isinstance(save_path, list):
                if len(save_path) != len(model):
                    raise ValueError(
                        f"Number of save paths ({len(save_path)}) must match "
                        f"number of models ({len(model)})"
                    )
                save_paths = save_path
            else:
                # Single path provided, create paths with index suffix
                base_path = save_path
                base_name, ext = os.path.splitext(base_path)
                save_paths = [
                    f"{base_name}_{i}{ext}" for i in range(len(model))
                ]

            results = []
            for i, m in enumerate(model):
                result = cls.plot_vector_field(
                    m, xlim, ylim, n_grid, t, direction, None, save_paths[i]
                )
                results.append(result)
            return results

        # Single model case
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Handle RealNVP separately
        if isinstance(model, RealNVP):
            return cls._plot_vector_field_realnvp(
                model, xlim, ylim, n_grid, direction, ax, save_path
            )

        # Get vector field from model once
        # At this point, model is VectorField2D, NeuralODE, or CNF
        vf = model if isinstance(model, VectorField2D) else model.vf

        # Convert t to tensor and extract scalar value for title
        t_value = float(t)
        t_tensor = torch.tensor(
            t_value, dtype=torch.float64, device=device
        )

        # Create grid
        x = np.linspace(xlim[0], xlim[1], n_grid)
        y = np.linspace(ylim[0], ylim[1], n_grid)
        X, Y = np.meshgrid(x, y)

        # Convert to tensor
        grid_points = torch.tensor(
            np.stack([X.ravel(), Y.ravel()], axis=1),
            dtype=torch.float64,
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

        # Save figure if path is provided
        if save_path is not None:
            fig = ax.figure
            # Ensure directory exists
            dirname = (
                os.path.dirname(save_path)
                if os.path.dirname(save_path) else '.'
            )
            os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return ax

    @classmethod
    def _plot_vector_field_realnvp(
        cls,
        flow: RealNVP,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        n_grid: int,
        direction: str,
        ax: Axes,
        save_path: Optional[str]
    ) -> Axes:
        """Helper method for RealNVP vector field visualization."""
        # Get the distribution from the flow (unconditional)
        flow.eval()
        device = next(flow.parameters()).device

        with torch.no_grad():
            dist = flow(None)
            transform = dist.transform

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

            if direction == 'forward':
                # Forward transformation: x -> z
                # Show displacement vectors from x to z
                z = transform(grid_points)
                displacement = z - grid_points
                title = 'RealNVP Forward Transform (x → z)'
            elif direction == 'inverse':
                # Inverse transformation: z -> x
                # Show displacement vectors from z to x
                x_transformed = transform.inv(grid_points)
                displacement = x_transformed - grid_points
                title = 'RealNVP Inverse Transform (z → x)'

            # Reshape displacement to grid
            displacement_np = displacement.cpu().numpy()
            U = displacement_np[:, 0].reshape(X.shape)
            V = displacement_np[:, 1].reshape(Y.shape)

            # Normalize for visualization
            magnitude = np.sqrt(U**2 + V**2)
            U_norm = U / (magnitude + 1e-8)
            V_norm = V / (magnitude + 1e-8)

            # Plot vector field
            ax.quiver(
                X, Y, U_norm, V_norm, magnitude,
                cmap='viridis', scale=20, width=0.005, alpha=0.7
            )

        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Save figure if path is provided
        if save_path is not None:
            fig = ax.figure
            # Ensure directory exists
            dirname = (
                os.path.dirname(save_path)
                if os.path.dirname(save_path) else '.'
            )
            os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return ax

    @classmethod
    def plot_transformation(
        cls,
        model: Union[
            NeuralODE,
            CNF,
            FFJORD,
            RealNVP,
            VectorField2D,
            List[Union[NeuralODE, CNF, FFJORD, RealNVP, VectorField2D]]
        ],
        n_samples: int = 20,
        n_steps: int = 100,
        xlim: Tuple[float, float] = (-3, 3),
        ylim: Tuple[float, float] = (-3, 3),
        axes: Optional[Tuple[Axes, Axes]] = None,
        save_path: Optional[Union[str, List[str]]] = None
    ) -> Union[Tuple[Axes, Axes], List[Tuple[Axes, Axes]]]:
        """Plot trajectories starting at z ~ N(0, I) and transforming to x.

        If a list of models is provided, creates a plot for each model.
        If save_path is a list, saves each plot to the corresponding path.

        Args:
            model: Single model or list of models.
            n_samples: Number of samples to generate.
            n_steps: Number of time steps.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            axes: Optional axes (only used for single model).
            save_path: Single path or list of paths for saving figures.

        Returns:
            Single tuple of axes or list of tuples of axes.
        """
        # Handle list of models
        if isinstance(model, list):
            if save_path is None:
                save_paths = [None] * len(model)
            elif isinstance(save_path, list):
                if len(save_path) != len(model):
                    raise ValueError(
                        f"Number of save paths ({len(save_path)}) must match "
                        f"number of models ({len(model)})"
                    )
                save_paths = save_path
            else:
                # Single path provided, create paths with index suffix
                base_path = save_path
                base_name, ext = os.path.splitext(base_path)
                save_paths = [
                    f"{base_name}_{i}{ext}" for i in range(len(model))
                ]

            results = []
            for i, m in enumerate(model):
                result = cls.plot_transformation(
                    m, n_samples, n_steps, xlim, ylim, None, save_paths[i]
                )
                results.append(result)
            return results

        # Single model case
        if axes is None:
            fig, (ax_left, ax_right) = plt.subplots(
                1, 2, figsize=(16, 8)
            )
        else:
            ax_left, ax_right = axes

        # Handle RealNVP separately
        if isinstance(model, RealNVP):
            return cls._plot_transformation_realnvp(
                model,
                n_samples,
                xlim,
                ylim,
                (ax_left, ax_right),
                save_path
            )

        # At this point, model is NeuralODE, CNF, or FFJORD
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
            x_t = model.sample(n_samples, n_steps)

        # Get initial and final states
        z_np = z.cpu().numpy()

        if x_t.dim() == 3:
            x_final_np = x_t[-1].cpu().numpy()  # [n_samples, 2]
        elif x_t.dim() == 2:
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
            marker='o',
            s=10,
            alpha=0.6,
            linewidths=1
        )

        # Unit circle (reference)
        circle = Circle(
            (0, 0),
            2,
            fill=False,
            linestyle='--',
            color='gray',
        )
        ax_left.add_patch(circle)

        ax_left.set_xlabel(r'$x_1$')
        ax_left.set_ylabel(r'$x_2$')
        ax_left.set_title('Sample points from normal z ~ N(0,I)')
        ax_left.set_xlim(xlim)
        ax_left.set_ylim(ylim)
        ax_left.grid(True, alpha=0.3)
        ax_left.axis('equal')

        ax_right.scatter(
            x_final_np[:, 0],
            x_final_np[:, 1],
            alpha=0.5,
            s=10
        )

        ax_right.set_xlabel(r'$x_1$')
        ax_right.set_ylabel(r'$x_2$')
        ax_right.set_title('Final state x')
        ax_right.set_xlim(xlim)
        ax_right.set_ylim(ylim)
        ax_right.grid(True, alpha=0.3)
        ax_right.axis('equal')

        # Save figure if path is provided
        if save_path is not None:
            fig = ax_left.figure
            # Ensure directory exists
            dirname = (
                os.path.dirname(save_path)
                if os.path.dirname(save_path) else '.'
            )
            os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return ax_left, ax_right

    @classmethod
    def _plot_transformation_realnvp(
        cls,
        flow: RealNVP,
        n_samples: int,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        axes: Optional[Tuple[Axes, Axes]],
        save_path: Optional[str]
    ) -> Tuple[Axes, Axes]:
        """Helper method for RealNVP transformation visualization."""
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
            marker='o',
            s=10,
            alpha=0.6,
            linewidths=1
        )

        # Unit circle (reference)
        circle = Circle(
            (0, 0), 2,
            fill=False,
            linestyle='--',
            color='gray',
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

        # Save figure if path is provided
        if save_path is not None:
            fig = ax_left.figure
            # Ensure directory exists
            dirname = (
                os.path.dirname(save_path)
                if os.path.dirname(save_path) else '.'
            )
            os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return ax_left, ax_right

    @classmethod
    def plot_transformation_gif(
        cls,
        model: Union[NeuralODE, CNF],
        dataset,
        n_steps: int = 100,
        n_samples: Optional[int] = None,
        xlim: Tuple[float, float] = (-3, 3),
        ylim: Tuple[float, float] = (-3, 3),
        save_path: Optional[str] = None,
        fps: int = 10
    ) -> None:
        """Generate a GIF showing data distribution transformation over time.

        Creates an animation showing how the data distribution evolves from
        x(0) to x(1) as the ODE is integrated forward.

        Args:
            model (Union[NeuralODE, CNF]): Model to use for trajectory
                integration.

            dataset: Dataset with .data attribute containing initial points.

            n_steps (int): Number of time steps in the trajectory.
                Default is 100.

            n_samples (int, optional): Number of samples to visualize.
                If None, uses all data points. Default is None.

            xlim (Tuple[float, float]): Limits in x direction.
                Default is (-3, 3).

            ylim (Tuple[float, float]): Limits in y direction.
                Default is (-3, 3).

            save_path (str, optional): Path to save the GIF.
                If None, raises ValueError. Default is None.

            fps (int): Frames per second for the GIF. Default is 10.

        Raises:
            ValueError: If save_path is None.
        """
        if save_path is None:
            raise ValueError("save_path must be provided to save the GIF")

        # Ensure directory exists
        dirname = (
            os.path.dirname(save_path)
            if os.path.dirname(save_path) else '.'
        )
        os.makedirs(dirname, exist_ok=True)

        # Get initial data
        data = dataset.data
        device = next(model.vf.parameters()).device

        # Sample subset if requested
        if n_samples is not None and n_samples < data.shape[0]:
            n_total = data.shape[0]
            indices = torch.randperm(n_total)[:n_samples]
            x_0 = data[indices].to(device)
        else:
            x_0 = data.to(device)

        # Integrate ODE forward from x(0) to x(1)
        model.eval()
        with torch.no_grad():
            x_t = model(x_0, n_steps)

        # x_t shape: (n_steps, n_samples, 2) or (n_steps+1, n_samples, 2)
        # Ensure we have the right shape
        if x_t.dim() == 3:
            # Shape is (n_steps, n_samples, 2) or (n_steps+1, n_samples, 2)
            n_frames = x_t.shape[0]
        else:
            raise ValueError(
                f"Unexpected x_t shape: {x_t.shape}. "
                f"Expected 3D [n_steps, n_samples, 2]"
            )

        # Create frames
        frames = []
        print(f"Generating {n_frames} frames...")

        for i in range(n_frames):
            # Get data at this time step
            x_t_step = x_t[i]  # (n_samples, 2)
            x_t_np = x_t_step.detach().cpu().numpy()

            # Create figure for this frame
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            # Plot data points
            ax.scatter(
                x_t_np[:, 0],
                x_t_np[:, 1],
                alpha=0.5,
                s=10,
                c='blue'
            )

            # Add unit circle reference for final state
            if i == n_frames - 1:
                circle = Circle(
                    (0, 0),
                    2,
                    fill=False,
                    linestyle='--',
                    color='gray',
                    alpha=0.5
                )
                ax.add_patch(circle)

            # Calculate time value (assuming t goes from 0 to 1)
            t_value = i / (n_frames - 1) if n_frames > 1 else 0.0

            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.set_title(f'Data Distribution at t={t_value:.3f}')
            # Set fixed limits first
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            # Use set_aspect with adjustable='box' to maintain equal aspect
            # while respecting the fixed limits
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)

            # Convert figure to image
            fig.canvas.draw()
            # Get the RGBA buffer and convert to RGB
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)[:, :, :3]  # Remove alpha channel
            frames.append(frame)

            plt.close(fig)

            if (i + 1) % 10 == 0:
                print(f"  Frame {i + 1}/{n_frames}")

        # Save as GIF
        print(f"Saving GIF to: {save_path}")
        imageio.mimsave(save_path, frames, fps=fps, loop=0)
        print("GIF saved successfully!")

    @classmethod
    def plot_data_distribution(
        cls,
        data: torch.Tensor | np.ndarray,
        ax: Optional[Axes] = None,
        title: str = 'Data Distribution',
        save_path: Optional[str] = None
    ) -> Axes:
        """Plot data distribution.

        Args:
            data (torch.Tensor | np.ndarray): Data tensor or array with shape
                (n_samples, 2).

            ax (Axes): Matplotlib axis.

            title (str): Plot title.

            save_path (str, optional): Path to save the figure. If provided,
                saves the figure to the specified path. Default is None.

        Returns:
            Axes: Matplotlib axis.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Convert to numpy
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        else:
            data = np.array(data)

        # Plot data points
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

        # Save figure if path is provided
        if save_path is not None:
            fig = ax.figure
            # Ensure directory exists
            dirname = (
                os.path.dirname(save_path)
                if os.path.dirname(save_path) else '.'
            )
            os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return ax


class MNISTViz:
    """Visualization class for MNIST datasets."""
    pca = None  # Class attribute for PCA instance

    @classmethod
    def project_to_2d(
        cls,
        data: torch.Tensor,
        components: int = 2,
        fit_pca: bool = False
    ) -> torch.Tensor:
        """Project high-dimensional data to 2D using PCA.

        Args:
            data (torch.Tensor): High-dimensional data with shape
                (n_samples, features).
            components (int): Number of PCA components to use. Default is 2.
            fit_pca (bool): Whether to fit PCA on this data. Default is False.

        Returns:
            torch.Tensor: 2D projected data with shape (n_samples, 2).
        """
        data_np = data.cpu().numpy()

        if fit_pca or cls.pca is None:
            cls.pca = PCA(n_components=components)
            data_2d = cls.pca.fit_transform(data_np)
        else:
            data_2d = cls.pca.transform(data_np)

        return torch.tensor(data_2d, dtype=torch.float64)

    @classmethod
    def plot_trajectories(
        cls,
        model: Union[NeuralODE, CNF, VectorField2D],
        dataset,
        n_samples: int = 20,
        n_steps: int = 100,
        axes: Optional[Tuple[Axes, Axes]] = None,
        save_path: Optional[str] = None,
        fit_pca_on_data: bool = True
    ) -> Tuple[Axes, Axes]:
        """Plot trajectories projected to 2D for random dataset samples.

        Creates two plots:
        - Left: Initial state x(0) projected to 2D with label colors
        - Right: Final state x(1) projected to 2D with label colors

        Args:
            model (Union[NeuralODE, CNF, VectorField2D]): Model. NeuralODE or CNF
                for trajectory integration.

            dataset: Dataset with .data attribute. Labels optional.

            n_samples (int): Number of random samples to plot.
                Default is 20.

            n_steps (int): Number of points in trajectory.

            axes (Tuple[Axes, Axes], optional): Two matplotlib axes for
                left and right plots. If None, creates new figure.

            save_path (str, optional): Path to save the figure. If provided,
                saves the figure to the specified path. Default is None.

            fit_pca_on_data (bool): Whether to fit PCA on the initial data.
                Default is True.

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
        labels = getattr(dataset, 'labels', None)

        # Sample random subset
        n_total = data.shape[0]
        n_plot = min(n_samples, n_total)
        indices = torch.randperm(n_total)[:n_plot]

        # Get device and determine model type once
        if isinstance(model, VectorField2D):
            raise ValueError(
                "plot_trajectories requires NeuralODE or CNF model. "
                "VectorField2D alone cannot integrate trajectories."
            )
        # At this point, model is NeuralODE or CNF
        device = next(model.vf.parameters()).device
        is_neural_ode = isinstance(model, NeuralODE)

        x_0 = data[indices].to(device)
        sampled_labels = labels[indices] if labels is not None else None

        # Fit PCA on initial data if requested
        if fit_pca_on_data:
            _ = cls.project_to_2d(x_0, fit_pca=True)

        # Integrate ODE from x(0) to x(1)
        model.eval()
        with torch.no_grad():
            if is_neural_ode:
                x_t, _ = model(x_0, n_steps)
            else:
                x_t, _ = model(x_0, n_steps)

        # Get final state x(1)
        if x_t.dim() == 3:
            x_1 = x_t[-1]  # (n_plot, features)
        elif x_t.dim() == 2:
            x_1 = x_t  # Already final state
        else:
            raise ValueError(f"Unexpected x_t shape: {x_t.shape}")

        # Project to 2D
        x_0_2d = cls.project_to_2d(x_0, fit_pca=False)
        x_1_2d = cls.project_to_2d(x_1, fit_pca=False)

        x_0_np = x_0_2d.cpu().numpy()
        x_1_np = x_1_2d.cpu().numpy()

        # LEFT PLOT: Initial state projected to 2D
        ax_left.scatter(
                x_0_np[:, 0],
                x_0_np[:, 1],
                alpha=0.5,
                s=10,
            )

        ax_left.set_xlabel('PC1')
        ax_left.set_ylabel('PC2')
        ax_left.set_title('Initial State x(0)')
        ax_left.grid(True, alpha=0.3)
        ax_left.axis('equal')

        # RIGHT PLOT: Final state projected to 2D
        ax_right.scatter(
                x_1_np[:, 0],
                x_1_np[:, 1],
                alpha=0.5,
                s=10,
            )

        ax_right.set_xlabel('PC1')
        ax_right.set_ylabel('PC2')
        ax_right.set_title('Final State x(1)')
        ax_right.grid(True, alpha=0.3)
        ax_right.axis('equal')

        # Save figure if path is provided
        if save_path is not None:
            fig = ax_left.figure
            # Ensure directory exists
            dirname = (
                os.path.dirname(save_path)
                if os.path.dirname(save_path) else '.'
            )
            os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return ax_left, ax_right

    @classmethod
    def plot_vector_field(
        cls,
        model: Union[VectorField2D, NeuralODE, CNF, RealNVP],
        data_sample: Optional[torch.Tensor] = None,
        n_grid: int = 20,
        t: float = 0.5,
        direction: str = 'forward',
        ax: Optional[Axes] = None,
        save_path: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None
    ) -> Axes:
        """Plot vector field projected to 2D.

        For high-dimensional models, the vector field is projected to 2D
        using PCA fitted on a data sample.

        Args:
            model (Union[VectorField2D, NeuralODE, CNF, RealNVP]): Model.

            data_sample (torch.Tensor, optional): Sample data to fit PCA.
                If None and PCA not fitted, raises error. Shape should be
                (n_samples, features).

            n_grid (int): Number of grid points per dimension.

            t (float): Time to evaluate (for ODE models).

            direction (str): For RealNVP, 'forward' or 'inverse'.
                Default is 'forward'.

            ax (Axes): Matplotlib axis.

            save_path (str, optional): Path to save the figure. If provided,
                saves the figure to the specified path. Default is None.

            xlim (Tuple[float, float], optional): Limits in x direction.
                If None, inferred from data.

            ylim (Tuple[float, float], optional): Limits in y direction.
                If None, inferred from data.

        Returns:
            Axes: Matplotlib axis.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Fit PCA if needed
        if cls.pca is None:
            if data_sample is None:
                raise ValueError(
                    "PCA not fitted. Provide data_sample to fit PCA, "
                    "or call plot_trajectories first."
                )
            _ = cls.project_to_2d(data_sample, fit_pca=True)

        # Handle RealNVP separately
        if isinstance(model, RealNVP):
            # For RealNVP, we need to project the transformation
            # This is more complex and may not be very meaningful
            raise NotImplementedError(
                "RealNVP vector field visualization for MNIST is not "
                "yet implemented."
            )

        # Get vector field from model once
        # At this point, model is VectorField2D, NeuralODE, or CNF
        vf = model if isinstance(model, VectorField2D) else model.vf
        device = next(vf.parameters()).device

        # Determine grid limits
        if xlim is None or ylim is None:
            if data_sample is not None:
                data_2d = cls.project_to_2d(data_sample, fit_pca=False)
                data_np = data_2d.cpu().numpy()
                margin = 0.5
                xlim = (
                    data_np[:, 0].min() - margin,
                    data_np[:, 0].max() + margin
                )
                ylim = (
                    data_np[:, 1].min() - margin,
                    data_np[:, 1].max() + margin
                )
            else:
                xlim = (-3, 3)
                ylim = (-3, 3)

        # Create 2D grid
        x = np.linspace(xlim[0], xlim[1], n_grid)
        y = np.linspace(ylim[0], ylim[1], n_grid)
        X, Y = np.meshgrid(x, y)

        # Convert 2D grid points back to high-dimensional space via PCA inverse
        grid_points_2d = np.stack([X.ravel(), Y.ravel()], axis=1)
        # Note: This is an approximation since PCA is lossy
        grid_points_highd = cls.pca.inverse_transform(grid_points_2d)
        grid_points = torch.tensor(
            grid_points_highd,
            dtype=torch.float64,
            device=device
        )

        # Calculate vector field in high-dimensional space
        t_value = float(t)
        t_tensor = torch.tensor(
            t_value, dtype=torch.float64, device=device
        )

        with torch.no_grad():
            dx_dt_highd = vf(t_tensor, grid_points).cpu().numpy()

        # Project vector field back to 2D
        dx_dt_2d = cls.pca.transform(dx_dt_highd)

        # Reshape
        U = dx_dt_2d[:, 0].reshape(X.shape)
        V = dx_dt_2d[:, 1].reshape(Y.shape)

        # Normalize for visualization
        magnitude = np.sqrt(U**2 + V**2)
        U_norm = U / (magnitude + 1e-8)
        V_norm = V / (magnitude + 1e-8)

        # Plot
        ax.quiver(
            X, Y, U_norm, V_norm, magnitude,
            cmap='viridis', scale=20, width=0.005, alpha=0.7
        )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Vector Field at t={t_value:.2f}')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Save figure if path is provided
        if save_path is not None:
            fig = ax.figure
            # Ensure directory exists
            dirname = (
                os.path.dirname(save_path)
                if os.path.dirname(save_path) else '.'
            )
            os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return ax

    @classmethod
    def plot_transformation(
        cls,
        model: Union[NeuralODE, CNF, RealNVP],
        n_samples: int = 20,
        n_steps: int = 100,
        axes: Optional[Tuple[Axes, Axes]] = None,
        save_path: Optional[str] = None,
        data_sample: Optional[torch.Tensor] = None
    ) -> Tuple[Axes, Axes]:
        """Plot transformation from z ~ N(0, I) to x, projected to 2D.

        Creates two plots:
        - Left: Initial z samples projected to 2D (gray, no labels)
        - Right: Final transformed state x projected to 2D (gray, no labels)

        Args:
            model (Union[NeuralODE, CNF, RealNVP]): Model.

            n_samples (int): Number of samples to plot trajectories for.
                Default is 20.

            n_steps (int): Number of points in each trajectory
                (for ODE models). Default is 100.

            axes (Tuple[Axes, Axes], optional): Two matplotlib axes for
                left and right plots. If None, creates new figure.

            save_path (str, optional): Path to save the figure. If provided,
                saves the figure to the specified path. Default is None.

            data_sample (torch.Tensor, optional): Sample data to fit PCA.
                If None and PCA not fitted, will fit on transformed samples.
                Shape should be (n_samples, features).

        Returns:
            Tuple[Axes, Axes]: Left and right matplotlib axes.
        """
        if axes is None:
            fig, (ax_left, ax_right) = plt.subplots(
                1, 2, figsize=(16, 8)
            )
        else:
            ax_left, ax_right = axes

        # Handle RealNVP separately
        if isinstance(model, RealNVP):
            return cls._plot_transformation_realnvp_mnist(
                model, n_samples, axes, save_path, data_sample
            )

        # At this point, model is NeuralODE or CNF
        device = next(model.vf.parameters()).device
        features = model.vf.features

        # Sample z ~ N(0, I)
        z = torch.randn(n_samples, features, device=device)

        model.eval()
        with torch.no_grad():
            x_t = model(z, n_steps)

        # Get final state
        if x_t.dim() == 3:
            x_final = x_t[-1]  # (n_samples, features)
        elif x_t.dim() == 2:
            x_final = x_t  # Already final state
        else:
            raise ValueError(
                f"Unexpected x_t shape: {x_t.shape}. "
                f"Expected 2D [n_samples, features] or "
                "3D [n_steps, n_samples, features]"
            )

        # Fit PCA on transformed samples or provided data
        if cls.pca is None:
            if data_sample is not None:
                _ = cls.project_to_2d(data_sample, fit_pca=True)
            else:
                # Fit on transformed samples
                _ = cls.project_to_2d(x_final, fit_pca=True)

        # Project to 2D
        z_2d = cls.project_to_2d(z, fit_pca=False)
        x_final_2d = cls.project_to_2d(x_final, fit_pca=False)

        z_np = z_2d.cpu().numpy()
        x_final_np = x_final_2d.cpu().numpy()

        # LEFT PLOT: Initial z samples projected to 2D
        ax_left.scatter(
            z_np[:, 0],
            z_np[:, 1],
            marker='o',
            s=10,
            alpha=0.6,
            linewidths=1
        )

        ax_left.set_xlabel('PC1')
        ax_left.set_ylabel('PC2')
        ax_left.set_title(
            'Sample points from normal z ~ N(0,I)')
        ax_left.grid(True, alpha=0.3)
        ax_left.axis('equal')

        # RIGHT PLOT: Final transformed state projected to 2D
        ax_right.scatter(
            x_final_np[:, 0],
            x_final_np[:, 1],
            alpha=0.5,
            s=10
        )

        ax_right.set_xlabel('PC1')
        ax_right.set_ylabel('PC2')
        ax_right.set_title('Final state x')
        ax_right.grid(True, alpha=0.3)
        ax_right.axis('equal')

        # Save figure if path is provided
        if save_path is not None:
            fig = ax_left.figure
            # Ensure directory exists
            dirname = (
                os.path.dirname(save_path)
                if os.path.dirname(save_path) else '.'
            )
            os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return ax_left, ax_right

    @classmethod
    def _plot_transformation_realnvp_mnist(
        cls,
        flow: RealNVP,
        n_samples: int,
        axes: Optional[Tuple[Axes, Axes]],
        save_path: Optional[str],
        data_sample: Optional[torch.Tensor]
    ) -> Tuple[Axes, Axes]:
        """Helper method for RealNVP transformation visualization in MNIST."""
        if axes is None:
            fig, (ax_left, ax_right) = plt.subplots(
                1, 2, figsize=(16, 8)
            )
        else:
            ax_left, ax_right = axes

        # Get the distribution from the flow (unconditional)
        flow.eval()
        device = next(flow.parameters()).device

        with torch.no_grad():
            dist = flow(None)

            # Sample z ~ N(0, I) from base distribution
            try:
                z = dist.base.sample((n_samples,))
            except AttributeError:
                # Fallback: sample manually from standard normal
                from torch.distributions import MultivariateNormal
                dim = dist.transform.domain.event_shape[0]
                base = MultivariateNormal(
                    torch.zeros(dim, device=device),
                    torch.eye(dim, device=device)
                )
                z = base.sample((n_samples,))

            # Transform z -> x using the inverse transform
            transform = dist.transform
            x = transform.inv(z)

        # Fit PCA if needed
        if cls.pca is None:
            if data_sample is not None:
                _ = cls.project_to_2d(data_sample, fit_pca=True)
            else:
                # Fit on transformed samples
                _ = cls.project_to_2d(x, fit_pca=True)

        # Project to 2D
        z_2d = cls.project_to_2d(z, fit_pca=False)
        x_2d = cls.project_to_2d(x, fit_pca=False)

        z_np = z_2d.cpu().numpy()
        x_np = x_2d.cpu().numpy()

        # LEFT PLOT: Initial z samples projected to 2D
        ax_left.scatter(
            z_np[:, 0],
            z_np[:, 1],
            marker='o',
            s=10,
            alpha=0.6,
            linewidths=1
        )

        ax_left.set_xlabel('PC1')
        ax_left.set_ylabel('PC2')
        ax_left.set_title('Sample points from normal z ~ N(0,I)')
        ax_left.grid(True, alpha=0.3)
        ax_left.axis('equal')

        # RIGHT PLOT: Final transformed state projected to 2D
        ax_right.scatter(
            x_np[:, 0],
            x_np[:, 1],
            alpha=0.5,
            s=10
        )

        ax_right.set_xlabel('PC1')
        ax_right.set_ylabel('PC2')
        ax_right.set_title('Transformed samples x = T⁻¹(z)')
        ax_right.grid(True, alpha=0.3)
        ax_right.axis('equal')

        # Save figure if path is provided
        if save_path is not None:
            fig = ax_left.figure
            # Ensure directory exists
            dirname = (
                os.path.dirname(save_path)
                if os.path.dirname(save_path) else '.'
            )
            os.makedirs(dirname, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        return ax_left, ax_right
