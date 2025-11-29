"""Visualization utilities."""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from src.models.neural_ode import NeuralODE
from src.models.vector_field import VectorField
from src.models.cnf import CNF
from typing import Optional, Tuple, Union
from zuko.flows import RealNVP
from sklearn.decomposition import PCA


class Synthetic2DViz:
    """Visualization class for 2D synthetic datasets."""
    @classmethod
    def plot_trajectories(
        cls,
        model: Union[NeuralODE, CNF, VectorField],
        dataset,
        n_samples: int = 20,
        t_span: Optional[torch.Tensor] = None,
        n_points: int = 100,
        axes: Optional[Tuple[Axes, Axes]] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Axes, Axes]:
        """Plot trajectories x(t) from x(0) to x(1) for random dataset samples.

        Creates two plots:
        - Left: Data distribution (transparent) + sampled x(0) points with
          label colors
        - Right: Final state x(1) with label colors

        Args:
            model (Union[NeuralODE, CNF, VectorField]): Model. NeuralODE or CNF
                for trajectory integration, or VectorField (will create
                NeuralODE wrapper).

            dataset: Dataset with .data and .labels attributes.

            n_samples (int): Number of random samples to plot.
                Default is 20.

            t_span (torch.Tensor): Time points for integration.

            n_points (int): Number of points in trajectory.

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
        labels = dataset.labels

        # Sample random subset
        n_total = data.shape[0]
        n_plot = min(n_samples, n_total)
        indices = torch.randperm(n_total)[:n_plot]

        # Get device and determine model type once
        if isinstance(model, VectorField):
            raise ValueError(
                "plot_trajectories requires NeuralODE or CNF model. "
                "VectorField alone cannot integrate trajectories."
            )
        # At this point, model is NeuralODE or CNF
        device = next(model.vf.parameters()).device
        is_neural_ode = isinstance(model, NeuralODE)

        x_0 = data[indices].to(device)
        sampled_labels = labels[indices]

        if t_span is None:
            t_span = torch.linspace(0, 1, n_points).to(device)

        # Integrate ODE from x(0) to x(1)
        model.eval()
        with torch.no_grad():
            if is_neural_ode:
                x_t, _ = model(x_0, t_span)
            else:
                # CNF forward goes from x to z, so we reverse
                x_t, _ = model(x_0, t_span, reverse=False)

        # Get final state x(1)
        if x_t.dim() == 3:
            x_1 = x_t[-1]  # (n_plot, features)
        elif x_t.dim() == 2:
            x_1 = x_t  # Already final state
        else:
            raise ValueError(f"Unexpected x_t shape: {x_t.shape}")

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
        model: Union[VectorField, NeuralODE, CNF, RealNVP],
        xlim: Tuple[float, float] = (-2, 2),
        ylim: Tuple[float, float] = (-2, 2),
        n_grid: int = 20,
        t: float = 0.5,
        direction: str = 'forward',
        ax: Optional[Axes] = None,
        save_path: Optional[str] = None
    ) -> Axes:
        """Plot vector field f(x, t) on a 2D grid.

        Args:
            model (Union[VectorField, NeuralODE, CNF, RealNVP]): Model.
                VectorField, NeuralODE, or CNF for ODE-based models.
                RealNVP for flow-based models.

            xlim (Tuple[float, float]): Limits in x direction.

            ylim (Tuple[float, float]): Limits in y direction.

            n_grid (int): Number of grid points.

            t (float): Time to evaluate (for ODE models).

            direction (str): For RealNVP, 'forward' or 'inverse'.
                Default is 'forward'.

            ax (Axes): Matplotlib axis.

            save_path (str, optional): Path to save the figure. If provided,
                saves the figure to the specified path. Default is None.

        Returns:
            Axes: Matplotlib axis.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Handle RealNVP separately
        if isinstance(model, RealNVP):
            return cls._plot_vector_field_realnvp(
                model, xlim, ylim, n_grid, direction, ax, save_path
            )

        # Get vector field from model once
        # At this point, model is VectorField, NeuralODE, or CNF
        vf = model if isinstance(model, VectorField) else model.vf
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
            else:
                raise ValueError(
                    f"direction must be 'forward' or 'inverse', got {direction}"
                )

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
        model: Union[NeuralODE, CNF, RealNVP],
        n_samples: int = 20,
        n_points: int = 100,
        xlim: Tuple[float, float] = (-3, 3),
        ylim: Tuple[float, float] = (-3, 3),
        axes: Optional[Tuple[Axes, Axes]] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Axes, Axes]:
        """Plot trajectories starting at z ~ N(0, I) and transforming to x.

        Creates two plots:
        - Left: Initial z samples from N(0, I) (gray, no labels)
        - Right: Final transformed state x (gray, no labels)
          Note: True class labels are unknown when transforming from z ~ N(0, I)

        Works with NeuralODE, CNF, and RealNVP models.

        Args:
            model (Union[NeuralODE, CNF, RealNVP]): Model.

            n_samples (int): Number of samples to plot trajectories for.
                Default is 20.

            n_points (int): Number of points in each trajectory
                (for ODE models). Default is 100.

            xlim (Tuple[float, float]): Limits in x direction.

            ylim (Tuple[float, float]): Limits in y direction.

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

        # Handle RealNVP separately
        if isinstance(model, RealNVP):
            return cls._plot_transformation_realnvp(
                model, n_samples, xlim, ylim, axes, save_path
            )

        # At this point, model is NeuralODE or CNF
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
                steps=n_points,
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

        ax_right.scatter(
            x_final_np[:, 0],
            x_final_np[:, 1],
            color='gray',
            alpha=0.5,
            s=10
        )

        ax_right.set_xlabel(r'$x_1$')
        ax_right.set_ylabel(r'$x_2$')
        ax_right.set_title('Final state x (transformed)')
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
    def plot_data_distribution(
        cls,
        data: torch.Tensor | np.ndarray,
        labels: Optional[torch.Tensor | np.ndarray] = None,
        ax: Optional[Axes] = None,
        title: str = 'Data Distribution',
        cmap: str = 'tab10',
        save_path: Optional[str] = None
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

            save_path (str, optional): Path to save the figure. If provided,
                saves the figure to the specified path. Default is None.

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

        return torch.tensor(data_2d, dtype=torch.float32)

    @classmethod
    def plot_trajectories(
        cls,
        model: Union[NeuralODE, CNF, VectorField],
        dataset,
        n_samples: int = 20,
        t_span: Optional[torch.Tensor] = None,
        n_points: int = 100,
        axes: Optional[Tuple[Axes, Axes]] = None,
        save_path: Optional[str] = None,
        fit_pca_on_data: bool = True
    ) -> Tuple[Axes, Axes]:
        """Plot trajectories projected to 2D for random dataset samples.

        Creates two plots:
        - Left: Initial state x(0) projected to 2D with label colors
        - Right: Final state x(1) projected to 2D with label colors

        Args:
            model (Union[NeuralODE, CNF, VectorField]): Model. NeuralODE or CNF
                for trajectory integration.

            dataset: Dataset with .data attribute. Labels optional.

            n_samples (int): Number of random samples to plot.
                Default is 20.

            t_span (torch.Tensor): Time points for integration.

            n_points (int): Number of points in trajectory.

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
        if isinstance(model, VectorField):
            raise ValueError(
                "plot_trajectories requires NeuralODE or CNF model. "
                "VectorField alone cannot integrate trajectories."
            )
        # At this point, model is NeuralODE or CNF
        device = next(model.vf.parameters()).device
        is_neural_ode = isinstance(model, NeuralODE)

        x_0 = data[indices].to(device)
        sampled_labels = labels[indices] if labels is not None else None

        if t_span is None:
            t_span = torch.linspace(0, 1, n_points).to(device)

        # Fit PCA on initial data if requested
        if fit_pca_on_data:
            _ = cls.project_to_2d(x_0, fit_pca=True)

        # Integrate ODE from x(0) to x(1)
        model.eval()
        with torch.no_grad():
            if is_neural_ode:
                x_t, _ = model(x_0, t_span)
            else:
                x_t, _ = model(x_0, t_span, reverse=False)

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

        if sampled_labels is not None:
            labels_np = sampled_labels.cpu().numpy()
            unique_labels = np.unique(labels_np)
            colors = plt.cm.get_cmap('tab10')(
                np.linspace(0, 1, len(unique_labels))
            )
            label_to_color = {
                label: colors[i] for i, label in enumerate(unique_labels)
            }
        else:
            labels_np = None

        # LEFT PLOT: Initial state projected to 2D
        if labels_np is not None:
            for label in unique_labels:
                mask = labels_np == label
                ax_left.scatter(
                    x_0_np[mask, 0],
                    x_0_np[mask, 1],
                    alpha=0.5,
                    s=10,
                    color=label_to_color[label],
                    label=f'Class {label}'
                )
            ax_left.legend()
        else:
            ax_left.scatter(
                x_0_np[:, 0],
                x_0_np[:, 1],
                alpha=0.5,
                s=10,
                color='gray'
            )

        ax_left.set_xlabel('PC1')
        ax_left.set_ylabel('PC2')
        ax_left.set_title('Initial State x(0) (2D projection)')
        ax_left.grid(True, alpha=0.3)
        ax_left.axis('equal')

        # RIGHT PLOT: Final state projected to 2D
        if labels_np is not None:
            for label in unique_labels:
                mask = labels_np == label
                ax_right.scatter(
                    x_1_np[mask, 0],
                    x_1_np[mask, 1],
                    alpha=0.5,
                    s=10,
                    color=label_to_color[label],
                    label=f'Class {label}'
                )
            ax_right.legend()
        else:
            ax_right.scatter(
                x_1_np[:, 0],
                x_1_np[:, 1],
                alpha=0.5,
                s=10,
                color='gray'
            )

        ax_right.set_xlabel('PC1')
        ax_right.set_ylabel('PC2')
        ax_right.set_title('Final State x(1) (2D projection)')
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
        model: Union[VectorField, NeuralODE, CNF, RealNVP],
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
            model (Union[VectorField, NeuralODE, CNF, RealNVP]): Model.

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
        # At this point, model is VectorField, NeuralODE, or CNF
        vf = model if isinstance(model, VectorField) else model.vf
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
            dtype=torch.float32,
            device=device
        )

        # Calculate vector field in high-dimensional space
        t_value = float(t)
        t_tensor = torch.tensor(
            t_value, dtype=torch.float32, device=device
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
        ax.set_title(f'Vector Field at t={t_value:.2f} (2D projection)')
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
        n_points: int = 100,
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

            n_points (int): Number of points in each trajectory
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
            t_span = torch.linspace(
                start=1.0,
                end=0.0,
                steps=n_points,
                device=device
            )
            x_t, _ = model(z, t_span, reverse=True)

        # Get final state
        if x_t.dim() == 3:
            x_final = x_t[-1]  # (n_samples, features)
        elif x_t.dim() == 2:
            x_final = x_t  # Already final state
        else:
            raise ValueError(
                f"Unexpected x_t shape: {x_t.shape}. "
                f"Expected 2D [n_samples, features] or 3D [n_steps, n_samples, features]"
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
            color='gray',
            marker='o',
            s=50,
            alpha=0.6,
            linewidths=1
        )

        ax_left.set_xlabel('PC1')
        ax_left.set_ylabel('PC2')
        ax_left.set_title('Sample points from normal z(0) ~ N(0,I) (2D projection)')
        ax_left.grid(True, alpha=0.3)
        ax_left.axis('equal')

        # RIGHT PLOT: Final transformed state projected to 2D
        ax_right.scatter(
            x_final_np[:, 0],
            x_final_np[:, 1],
            color='gray',
            alpha=0.5,
            s=10
        )

        ax_right.set_xlabel('PC1')
        ax_right.set_ylabel('PC2')
        ax_right.set_title('Final state x (transformed, 2D projection)')
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
            color='gray',
            marker='o',
            s=50,
            alpha=0.6,
            linewidths=1
        )

        ax_left.set_xlabel('PC1')
        ax_left.set_ylabel('PC2')
        ax_left.set_title('Sample points from normal z ~ N(0,I) (2D projection)')
        ax_left.grid(True, alpha=0.3)
        ax_left.axis('equal')

        # RIGHT PLOT: Final transformed state projected to 2D
        ax_right.scatter(
            x_np[:, 0],
            x_np[:, 1],
            color='gray',
            alpha=0.5,
            s=10
        )

        ax_right.set_xlabel('PC1')
        ax_right.set_ylabel('PC2')
        ax_right.set_title('Transformed samples x = T⁻¹(z) (2D projection)')
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
