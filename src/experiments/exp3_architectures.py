"""Experimento 3: Comparação de arquiteturas de Vector Field."""
import os
import csv
import time
import torch
import torch.nn as nn
import math
import torch.optim as optim
from src.models.vector_field import VectorField2D
from src.models.ffjord import FFJORD
from src.utils.datasets import Synthetic2D, get_dataloader
from src.utils.training import train_ffjord, count_nfe
from src.utils.visualization import Synthetic2DViz
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing import Dict, Tuple


# Custom VectorField2D variants with different time embeddings
class SimpleTimeVF(VectorField2D):
    """VectorField2D with simple scalar time concatenation (no embedding)."""

    def __init__(
        self,
        features: int,
        hidden_dims: list[int] = [64, 64],
        time_embed_dim: int = 1  # Unused, kept for compatibility
    ) -> None:
        """Initialize SimpleTimeVF."""
        # Override to use features + 1 (for scalar t)
        super().__init__(features, hidden_dims, time_embed_dim=1)
        self.hidden_dims = hidden_dims
        # Rebuild network with features + 1 input
        dims = [features + 1] + hidden_dims + [features]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(
                nn.Linear(
                    dims[i],
                    dims[i + 1],
                    dtype=torch.float64
                )
            )
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        # Reinitialize last layer
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Simple scalar time (no embedding)."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        elif t.dim() != 1:
            raise ValueError(
                f"Time tensor must have shape [batch, ], got {t.shape}"
            )
        return t.unsqueeze(-1)  # (batch, 1)


class SinusoidalVF(VectorField2D):
    """VectorField2D with enhanced sinusoidal time embedding."""

    def __init__(
        self,
        features: int,
        hidden_dims: list[int] = [64, 64],
        time_embed_dim: int = 32  # Larger embedding for more frequencies
    ) -> None:
        """Initialize SinusoidalVF."""
        super().__init__(features, hidden_dims, time_embed_dim)
        self.hidden_dims = hidden_dims

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Enhanced sinusoidal embedding with multiple frequencies."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        elif t.dim() != 1:
            raise ValueError(
                f"Time tensor must have shape [batch, ], got {t.shape}"
            )

        # Use multiple frequency scales
        half = self.time_embed_dim // 2
        i = torch.arange(0, half, dtype=torch.float64, device=t.device)
        # Multiple frequency scales: 1, 10, 100, 1000, 10000
        freqs = torch.exp(
            -i * 2.0 * math.log(10000.0) / self.time_embed_dim
        )
        # Add additional frequencies
        freqs_extra = torch.tensor(
            [1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
            dtype=torch.float64,
            device=t.device
        )[:half]
        if len(freqs_extra) < half:
            freqs = torch.cat(
                [freqs_extra, freqs[:half - len(freqs_extra)]]
            )
        else:
            freqs = freqs_extra[:half]

        return torch.cat([
            torch.sin(t.unsqueeze(-1) * freqs),
            torch.cos(t.unsqueeze(-1) * freqs)
        ], dim=-1)


class LearnableTimeVF(VectorField2D):
    """VectorField2D with learnable time embedding via small network."""

    def __init__(
        self,
        features: int,
        hidden_dims: list[int] = [64, 64],
        time_embed_dim: int = 16
    ) -> None:
        """Initialize LearnableTimeVF."""
        super().__init__(features, hidden_dims, time_embed_dim)
        self.hidden_dims = hidden_dims
        # Learnable time embedding network
        self.time_net = nn.Sequential(
            nn.Linear(1, 32, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(32, time_embed_dim, dtype=torch.float64),
            nn.Tanh()
        )

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Learnable time embedding."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        elif t.dim() != 1:
            raise ValueError(
                f"Time tensor must have shape [batch, ], got {t.shape}"
            )
        t_input = t.unsqueeze(-1)  # (batch, 1)
        return self.time_net(t_input)  # (batch, time_embed_dim)


def get_checkpoint_path(
    config_name: str,
    checkpoint_dir: str = 'results/checkpoints/exp3'
) -> str:
    """Generate checkpoint file path for a configuration."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Create safe filename
    safe_name = (
        config_name.replace(' ', '_')
        .replace('=', '_')
        .replace(',', '')
    )
    filename = f"checkpoint_{safe_name}.pt"
    return os.path.join(checkpoint_dir, filename)


def save_checkpoint(
    model: FFJORD,
    optimizer: optim.Optimizer,
    history: Dict[str, list],
    metrics: Dict[str, float],
    config_name: str,
    checkpoint_path: str,
    hidden_dims: list[int]
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'metrics': metrics,
        'config_name': config_name,
        'model_config': {
            'features': model.vf.features,
            'hidden_dims': hidden_dims,
            'time_embed_dim': model.vf.time_embed_dim,
            'vf_type': type(model.vf).__name__,
            'num_samples': model.num_samples
        }
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint loaded from: {checkpoint_path}")
    return checkpoint


def compute_reconstruction_error(
    model: FFJORD,
    x: torch.Tensor,
    n_steps: int = 100
) -> float:
    """Compute reconstruction error: forward then backward."""
    model.eval()
    with torch.no_grad():
        # Forward: x -> z
        z, _ = model(x, n_steps)

        # Backward: z -> x_recon
        x_recon, _ = model.backward(z, n_steps)

        # Reconstruction error
        error = ((x - x_recon) ** 2).mean().item()
    return error


def train_ffjord_with_metrics(
    model: FFJORD,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    n_epochs: int = 50,
    convergence_threshold: float = 0.001,
    convergence_window: int = 5
) -> Tuple[Dict[str, list], Dict[str, float]]:
    """Train FFJORD and track metrics.

    Returns:
        history: Training history (losses, test_recon_error, test_log_prob)
        metrics: Final metrics (test_recon_error, test_log_prob, training_time,
            nfe, convergence_epoch, final_loss)
    """
    model.train()
    start_time = time.time()

    history = {
        'loss': [],
        'test_recon_error': [],
        'test_log_prob': []
    }

    best_loss = float('inf')
    convergence_epoch = None
    loss_window = []

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_nll = 0.0
        n_batches = 0

        for x0 in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}"
        ):
            optimizer.zero_grad()
            x0 = x0.to(device)

            # Calculate log-likelihood
            log_prob = model.log_prob(x0)
            nll = -log_prob.mean()
            loss = nll

            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            total_nll += nll.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Evaluate on test set
        model.eval()
        test_recon_errors = []
        test_log_probs = []
        with torch.no_grad():
            for x_test in test_loader:
                x_test = x_test.to(device)
                recon_error = compute_reconstruction_error(model, x_test)
                test_recon_errors.append(recon_error)
                log_prob = model.log_prob(x_test).mean().item()
                test_log_probs.append(log_prob)
        avg_test_recon_error = sum(test_recon_errors) / len(test_recon_errors)
        avg_test_log_prob = sum(test_log_probs) / len(test_log_probs)
        model.train()

        history['loss'].append(avg_loss)
        history['test_recon_error'].append(avg_test_recon_error)
        history['test_log_prob'].append(avg_test_log_prob)

        # Check convergence
        loss_window.append(avg_loss)
        if len(loss_window) > convergence_window:
            loss_window.pop(0)

        if len(loss_window) == convergence_window:
            loss_change = (
                abs(loss_window[-1] - loss_window[0]) / loss_window[0]
            )
            if (loss_change < convergence_threshold and
                    convergence_epoch is None):
                convergence_epoch = epoch + 1

        if avg_loss < best_loss:
            best_loss = avg_loss

        print(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}, "
            f"Test recon error: {avg_test_recon_error:.6f}, "
            f"Test log-prob: {avg_test_log_prob:.4f}"
        )

    training_time = time.time() - start_time

    # Count NFEs
    sample_batch = torch.randn(10, 2).to(device)
    nfe = count_nfe(model, sample_batch)

    # Final test metrics
    model.eval()
    final_test_recon_errors = []
    final_test_log_probs = []
    with torch.no_grad():
        for x_test in test_loader:
            x_test = x_test.to(device)
            recon_error = compute_reconstruction_error(model, x_test)
            final_test_recon_errors.append(recon_error)
            log_prob = model.log_prob(x_test).mean().item()
            final_test_log_probs.append(log_prob)
    final_test_recon_error = (
        sum(final_test_recon_errors) / len(final_test_recon_errors)
    )
    final_test_log_prob = (
        sum(final_test_log_probs) / len(final_test_log_probs)
    )
    final_loss = -final_test_log_prob

    metrics = {
        'test_recon_error': final_test_recon_error,
        'test_log_prob': final_test_log_prob,
        'final_loss': final_loss,
        'training_time': training_time,
        'nfe': nfe,
        'convergence_epoch': (
            convergence_epoch if convergence_epoch else n_epochs
        )
    }

    return history, metrics


def compare_architectures(
    checkpoint_dir: str = 'results/checkpoints/exp3',
    resume: bool = True,
    n_epochs: int = 50
) -> Dict[str, Dict]:
    """Compara diferentes arquiteturas de Vector Field.

    Args:
        checkpoint_dir: Directory to save/load checkpoints.
        resume: If True, load existing checkpoints instead of retraining.
        n_epochs: Number of training epochs if training from scratch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    full_dataset = Synthetic2D(
        n_samples=5000, noise=0.05, dataset_type='moons'
    )
    # Split into train and test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = get_dataloader(
        train_dataset, batch_size=128, shuffle=True
    )
    test_loader = get_dataloader(
        test_dataset, batch_size=128, shuffle=False
    )

    # Architecture configurations
    # 4 depth/width configs × 3 time embeddings = 12 configs
    depth_configs = {
        'Shallow': [64, 64],
        'Medium': [64, 64, 64, 64],
        'Deep': [128, 128, 128, 128, 128, 128],
        'Wide': [256, 256]
    }

    time_embed_configs = {
        'Simple': (SimpleTimeVF, 1),
        'Sinusoidal': (SinusoidalVF, 32),
        'Learnable': (LearnableTimeVF, 16)
    }

    results = {}

    for depth_name, hidden_dims in depth_configs.items():
        for (
            time_name,
            (vf_class, time_embed_dim)
        ) in time_embed_configs.items():
            config_name = f"{depth_name}_{time_name}"
            print(f"\n=== Testando: {config_name} ===")

            checkpoint_path = get_checkpoint_path(config_name, checkpoint_dir)

            # Try to load checkpoint if it exists and resume is enabled
            if resume and os.path.exists(checkpoint_path):
                print(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = load_checkpoint(checkpoint_path, device)

                # Reconstruct model
                model_config = checkpoint['model_config']
                vf_class_loaded = globals()[model_config['vf_type']]
                vf = vf_class_loaded(
                    features=model_config['features'],
                    hidden_dims=model_config['hidden_dims'],
                    time_embed_dim=model_config['time_embed_dim']
                )
                model = FFJORD(
                    vf,
                    num_samples=model_config.get('num_samples', 1)
                ).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])

                # Reconstruct optimizer
                optimizer = optim.Adam(model.parameters(), lr=1e-4)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Load results
                history = checkpoint['history']
                metrics = checkpoint['metrics']

                print(
                    f"Loaded: Test recon error="
                    f"{metrics['test_recon_error']:.6f}, "
                    f"NFEs={metrics['nfe']}, "
                    f"Training time={metrics['training_time']:.2f}s"
                )
            else:
                # Train from scratch
                vf = vf_class(
                    features=2,
                    hidden_dims=hidden_dims,
                    time_embed_dim=time_embed_dim
                )
                model = FFJORD(vf, num_samples=1).to(device)
                optimizer = optim.Adam(model.parameters(), lr=1e-4)

                # Train with metrics
                history, metrics = train_ffjord_with_metrics(
                    model,
                    train_loader,
                    test_loader,
                    optimizer,
                    device,
                    n_epochs=n_epochs
                )

                # Save checkpoint
                save_checkpoint(
                    model, optimizer, history, metrics,
                    config_name, checkpoint_path, hidden_dims
                )

            # Plot vector field
            plot_dir = 'results/plots/exp3'
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(
                plot_dir, f"vector_field_{config_name}.png"
            )
            Synthetic2DViz.plot_vector_field(
                model, xlim=(-2, 2), ylim=(-2, 2),
                n_grid=20, t=0.5, save_path=plot_path
            )

            results[config_name] = {
                'model': model,
                'history': history,
                'metrics': metrics,
                'config': {
                    'depth': depth_name,
                    'time_embed': time_name,
                    'hidden_dims': hidden_dims
                }
            }

    return results


def save_summary_csv(
    results: Dict[str, Dict],
    save_dir: str = 'results'
) -> None:
    """Save architecture comparison results to CSV file."""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'exp3_metrics.csv')

    # Prepare CSV data
    rows = []
    for config_name, result in results.items():
        metrics = result['metrics']
        config = result['config']
        rows.append({
            'Config': config_name,
            'Depth': config['depth'],
            'Time_Embed': config['time_embed'],
            'Hidden_Dims': str(config['hidden_dims']),
            'Loss': metrics.get('final_loss', -metrics.get('test_log_prob', 0)),
            'NFE': metrics['nfe'],
            'Test_Log_Prob': metrics.get('test_log_prob', 0),
            'Test_Recon_Error': metrics['test_recon_error'],
            'Training_Time': metrics['training_time'],
            'Convergence_Epoch': metrics['convergence_epoch']
        })

    # Write CSV
    fieldnames = [
        'Config', 'Depth', 'Time_Embed', 'Hidden_Dims', 'Loss', 'NFE',
        'Test_Log_Prob', 'Test_Recon_Error', 'Training_Time', 'Convergence_Epoch'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV metrics saved to: {csv_path}")


if __name__ == '__main__':
    # Run comparison
    results = compare_architectures(n_epochs=50)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for config_name, result in sorted(
        results.items(),
        key=lambda x: x[1]['metrics'].get('final_loss', float('inf'))
    ):
        metrics = result['metrics']
        print(
            f"{config_name}: "
            f"Loss={metrics.get('final_loss', -metrics.get('test_log_prob', 0)):.4f}, "
            f"Test recon error={metrics['test_recon_error']:.6f}, "
            f"Time={metrics['training_time']:.2f}s, "
            f"NFEs={metrics['nfe']}, "
            f"Convergence={metrics['convergence_epoch']}"
        )

    # Save comprehensive summary
    save_summary_csv(results)

    # Plot samples from all models for comparison
    print("\n" + "=" * 60)
    print("GENERATING SAMPLE COMPARISON PLOT")
    print("=" * 60)
    models_dict = {
        config_name: result['model']
        for config_name, result in results.items()
    }
    plot_dir = os.path.join('results', 'plots', 'exp3')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'samples_comparison.png')
    Synthetic2DViz.plot_samples(
        models=models_dict,
        n_samples=1000,
        n_steps=100,
        save_path=plot_path
    )
    print(f"Sample comparison plot saved to: {plot_path}")
