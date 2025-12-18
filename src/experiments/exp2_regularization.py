"""Experimento 2: Regularização e tolerâncias."""
import os
import csv
import torch
import torch.optim as optim
from src.models.ffjord import FFJORD
from src.models.vector_field import VectorField2D
from src.utils.datasets import Synthetic2D, get_dataloader
from src.utils.training import count_nfe
from src.utils.visualization import Synthetic2DViz
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict


def compute_regularizations(
    vf: VectorField2D,
    x: torch.Tensor,
    t: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Computa termos de regularização."""
    x = x.requires_grad_(True)
    v = vf(t, x)

    # R1: Kinetic Energy
    # Penaliza velocidades altas: E[||v||²]
    kinetic_energy = (v ** 2).sum(dim=-1).mean()

    # R2: Jacobian Frobenius Norm
    # Penaliza Jacobian complexo: E[||∂v/∂x||_F²]
    jac_frob = 0.0
    for i in range(v.shape[1]):
        grad_v_i = torch.autograd.grad(
            v[:, i].sum(), x,
            create_graph=True, retain_graph=True
        )[0]
        jac_frob += (grad_v_i ** 2).sum()
    jac_frob = jac_frob / v.shape[0]  # Average over batch

    return {
        'kinetic_energy': kinetic_energy,
        'jacobian_frobenius': jac_frob
    }


def train_ffjord_with_regularization(
    model: FFJORD,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 100,
    lambda_ke: float = 0.0,
    lambda_jf: float = 0.0,
    reg_time: float = 0.5
) -> Dict[str, list]:
    """Train FFJORD with regularization terms.

    Args:
        model: FFJORD model.
        dataloader: DataLoader for training data.
        optimizer: Optimizer for training.
        device: Device to run training on.
        n_epochs: Number of training epochs.
        lambda_ke: Weight for kinetic energy regularization.
        lambda_jf: Weight for Jacobian Frobenius regularization.
        reg_time: Time point at which to compute regularizations.

    Returns:
        Dictionary with training history (losses, ke, jf).
    """
    model.train()

    history = {
        'loss': [],
        'nll': [],
        'ke': [],
        'jf': []
    }

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_nll = 0.0
        total_ke = 0.0
        total_jf = 0.0
        n_batches = 0

        for x0 in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}"
        ):
            optimizer.zero_grad()
            x0 = x0.to(device)

            # Calculate log-likelihood
            log_prob = model.log_prob(x0)
            nll = -log_prob.mean()

            # Compute regularizations at time reg_time
            t_reg = torch.tensor(reg_time, device=device)
            regs = compute_regularizations(model.vf, x0, t_reg)
            ke = regs['kinetic_energy']
            jf = regs['jacobian_frobenius']

            # Total loss: NLL + regularization terms
            loss = nll + lambda_ke * ke + lambda_jf * jf

            loss.backward()

            # Gradient clipping (optional, but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            total_nll += nll.item()
            total_ke += ke.item()
            total_jf += jf.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_nll = total_nll / n_batches
        avg_ke = total_ke / n_batches
        avg_jf = total_jf / n_batches

        history['loss'].append(avg_loss)
        history['nll'].append(avg_nll)
        history['ke'].append(avg_ke)
        history['jf'].append(avg_jf)

        print(
            f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, "
            f"NLL: {avg_nll:.4f}, KE: {avg_ke:.4f}, JF: {avg_jf:.4f}"
        )

    return history


def get_checkpoint_path(
    lambda_ke: float,
    lambda_jf: float,
    checkpoint_dir: str = 'results/checkpoints/exp2'
) -> str:
    """Generate checkpoint file path for a regularization configuration.

    Args:
        lambda_ke: Kinetic energy regularization weight.
        lambda_jf: Jacobian Frobenius regularization weight.
        checkpoint_dir: Directory to save checkpoints.

    Returns:
        Full path to checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Create safe filename by replacing special characters
    safe_ke = str(lambda_ke).replace('.', '_')
    safe_jf = str(lambda_jf).replace('.', '_')
    filename = f"checkpoint_ke_{safe_ke}_jf_{safe_jf}.pt"
    return os.path.join(checkpoint_dir, filename)


def save_checkpoint(
    model: FFJORD,
    optimizer: torch.optim.Optimizer,
    history: Dict[str, list],
    nfe: int,
    final_log_prob: float,
    lambda_ke: float,
    lambda_jf: float,
    checkpoint_path: str
) -> None:
    """Save model checkpoint.

    Args:
        model: Trained FFJORD model.
        optimizer: Optimizer state.
        history: Training history dictionary.
        nfe: Number of function evaluations.
        final_log_prob: Final log-likelihood.
        lambda_ke: Kinetic energy regularization weight.
        lambda_jf: Jacobian Frobenius regularization weight.
        checkpoint_path: Path to save checkpoint.
    """
    final_loss = -final_log_prob
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'nfe': nfe,
        'final_log_prob': final_log_prob,
        'final_loss': final_loss,
        'lambda_ke': lambda_ke,
        'lambda_jf': lambda_jf,
        'model_config': {
            'features': model.vf.features,
            'hidden_dims': [64, 64],  # Hardcoded for now, match training
            'time_embed_dim': 16
        }
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device
) -> Dict:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        Dictionary with checkpoint data including model, optimizer, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint loaded from: {checkpoint_path}")
    return checkpoint


def compare_regularizations(
    checkpoint_dir: str = 'results/checkpoints/exp2',
    resume: bool = True,
    n_epochs: int = 100
):
    """Compara diferentes combinações de regularização.

    Args:
        checkpoint_dir: Directory to save/load checkpoints.
        resume: If True, load existing checkpoints instead of retraining.
        n_epochs: Number of training epochs if training from scratch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = Synthetic2D(n_samples=5000, noise=0.05, dataset_type='moons')
    dataloader = get_dataloader(dataset, batch_size=128, shuffle=True)

    # Configurações de regularização para comparar
    # (lambda_ke, lambda_jf)
    regularization_configs = [
        (0.0, 0.0),      # Sem regularização (baseline)
        (0.01, 0.0),     # Apenas Kinetic Energy
        (0.0, 0.01),     # Apenas Jacobian Frobenius
        (0.01, 0.01),    # Ambas regularizações
        (0.1, 0.0),      # KE mais forte
        (0.0, 0.1),      # JF mais forte
        (0.1, 0.1),     # Ambas mais fortes
    ]

    results = {}

    for lambda_ke, lambda_jf in regularization_configs:
        config_name = f"λ_KE={lambda_ke}, λ_JF={lambda_jf}"
        print(f"\n=== Testando {config_name} ===")

        # Checkpoint path for this configuration
        checkpoint_path = get_checkpoint_path(
            lambda_ke, lambda_jf, checkpoint_dir
        )

        # Try to load checkpoint if it exists and resume is enabled
        if resume and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path, device)

            # Reconstruct model
            model_config = checkpoint['model_config']
            vf = VectorField2D(
                features=model_config['features'],
                hidden_dims=model_config['hidden_dims'],
                time_embed_dim=model_config['time_embed_dim']
            )
            model = FFJORD(vf).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Reconstruct optimizer
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load results
            history = checkpoint['history']
            nfe = checkpoint['nfe']
            final_log_prob = checkpoint['final_log_prob']
            final_loss = checkpoint.get('final_loss', -final_log_prob)

            print(
                f"Loaded: NFEs={nfe}, Loss={final_loss:.4f}, "
                f"log-prob={final_log_prob:.4f}"
            )
        else:
            # Train from scratch
            # Modelo
            vf = VectorField2D(
                features=2, hidden_dims=[64, 64], time_embed_dim=16
            )
            model = FFJORD(vf).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            # Treinar com regularização
            history = train_ffjord_with_regularization(
                model,
                dataloader,
                optimizer,
                device,
                n_epochs=n_epochs,
                lambda_ke=lambda_ke,
                lambda_jf=lambda_jf
            )

            # Contar NFEs (usando amostras de N(0, I))
            sample_batch = torch.randn(10, 2).to(device)
            nfe = count_nfe(model, sample_batch)

            # Calcular log-likelihood final no dataset
            model.eval()
            with torch.no_grad():
                test_batch = torch.stack(
                    [dataset[i] for i in range(min(100, len(dataset)))]
                ).to(device)
                final_log_prob = model.log_prob(test_batch).mean().item()
                final_loss = -final_log_prob

            # Save checkpoint
            save_checkpoint(
                model, optimizer, history, nfe, final_log_prob,
                lambda_ke, lambda_jf, checkpoint_path
            )

            print(
                f"NFEs: {nfe}, Loss: {final_loss:.4f}, "
                f"Final log-prob: {final_log_prob:.4f}"
            )

        results[config_name] = {
            'lambda_ke': lambda_ke,
            'lambda_jf': lambda_jf,
            'nfe': nfe,
            'final_log_prob': final_log_prob,
            'final_loss': final_loss,
            'history': history,
            'model': model
        }

    return results


def compute_convergence_epoch(
    history: Dict[str, list],
    convergence_threshold: float = 0.001,
    convergence_window: int = 5
) -> int:
    """Compute convergence epoch from training history.

    Args:
        history: Training history with 'loss' key.
        convergence_threshold: Relative change threshold for convergence.
        convergence_window: Number of epochs to check for stability.

    Returns:
        Epoch number when convergence was detected, or last epoch if not
        converged.
    """
    if 'loss' not in history or len(history['loss']) < convergence_window:
        return len(history.get('loss', []))

    loss_list = history['loss']
    for i in range(convergence_window - 1, len(loss_list)):
        window_start = i - convergence_window + 1
        window_losses = loss_list[window_start:i + 1]
        if len(window_losses) == convergence_window:
            loss_change = abs(
                window_losses[-1] - window_losses[0]
            ) / (window_losses[0] + 1e-8)
            if loss_change < convergence_threshold:
                return i + 1  # Return epoch number (1-indexed)

    return len(loss_list)  # Did not converge within training period


def save_summary_csv(
    results: Dict[str, Dict],
    save_dir: str = 'results'
) -> None:
    """Save regularization results to CSV summary table.

    Args:
        results: Dictionary with results from compare_regularizations().
        save_dir: Directory to save CSV file.
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'exp2_metrics.csv')

    # Get baseline for computing differences
    baseline = results.get('λ_KE=0.0, λ_JF=0.0')
    baseline_log_prob = baseline['final_log_prob'] if baseline else None
    baseline_nfe = baseline['nfe'] if baseline else None
    baseline_loss = (
        baseline.get('final_loss', -baseline_log_prob)
        if baseline else None
    )

    # Prepare CSV data
    rows = []
    for config_name, result in results.items():
        final_loss = result.get('final_loss', -result['final_log_prob'])
        ll_diff = (
            result['final_log_prob'] - baseline_log_prob
            if baseline_log_prob is not None
            else None
        )
        nfe_diff = (
            result['nfe'] - baseline_nfe
            if baseline_nfe is not None
            else None
        )
        loss_diff = (
            final_loss - baseline_loss
            if baseline_loss is not None
            else None
        )

        # Compute convergence epoch from history
        convergence_epoch = None
        if 'history' in result and result['history']:
            convergence_epoch = compute_convergence_epoch(result['history'])

        rows.append({
            'Config': config_name,
            'Lambda_KE': result['lambda_ke'],
            'Lambda_JF': result['lambda_jf'],
            'Loss': final_loss,
            'NFE': result['nfe'],
            'Log_Prob': result['final_log_prob'],
            'Loss_Diff': loss_diff if loss_diff is not None else '',
            'Log_Prob_Diff': ll_diff if ll_diff is not None else '',
            'NFE_Diff': nfe_diff if nfe_diff is not None else '',
            'Convergence_Epoch': (
                convergence_epoch if convergence_epoch is not None else ''
            )
        })

    # Write CSV
    fieldnames = [
        'Config', 'Lambda_KE', 'Lambda_JF', 'Loss', 'NFE', 'Log_Prob',
        'Loss_Diff', 'Log_Prob_Diff', 'NFE_Diff', 'Convergence_Epoch'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV metrics saved to: {csv_path}")


def analyze_regularization_impact(
    results: Dict[str, Dict],
    dataset: Synthetic2D,
    save_dir: str = 'results'
) -> None:
    """Generate CSV summary of regularization impact.

    Args:
        results: Dictionary with results from compare_regularizations().
        dataset: Original dataset for reference (unused, kept for
            compatibility).
        save_dir: Directory to save CSV summary.
    """
    print("\n" + "=" * 60)
    print("GENERATING CSV SUMMARY")
    print("=" * 60)

    save_summary_csv(results, save_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = Synthetic2D(n_samples=5000, noise=0.05, dataset_type='moons')

    # Run comparison
    results = compare_regularizations()

    # Print summary
    print("\n=== Resumo ===")
    for config_name, result in results.items():
        print(
            f"{config_name}: "
            f"NFEs={result['nfe']}, "
            f"log-prob={result['final_log_prob']:.4f}"
        )

    # Generate comprehensive analysis
    analyze_regularization_impact(results, dataset)

    # Plot transformations from all models for comparison
    print("\n" + "=" * 60)
    print("GENERATING TRANSFORMATION PLOTS")
    print("=" * 60)
    models_list = []
    save_paths = []

    for config_name, result in results.items():
        models_list.append(result['model'])

        # Create save path
        plot_dir = os.path.join('results', 'figures', 'exp2')
        os.makedirs(plot_dir, exist_ok=True)
        # Create safe filename
        safe_name = (
            config_name.replace(' ', '_')
            .replace('=', '_')
            .replace(',', '')
            .replace('λ', 'lambda')
        )
        plot_path = os.path.join(
            plot_dir, f'transformation_{safe_name}.png'
        )
        save_paths.append(plot_path)

    # Plot transformations for all configurations
    Synthetic2DViz.plot_transformation(
        models_list,
        n_samples=1000,
        n_steps=100,
        save_path=save_paths
    )
    print(f"Transformation plots saved to: {plot_dir}")
