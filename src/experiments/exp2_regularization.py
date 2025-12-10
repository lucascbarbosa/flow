"""Experimento 2: Regularização e tolerâncias."""
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from ..models.ffjord import FFJORD
from ..models.vector_field import VectorField2D
from ..utils.datasets import Synthetic2D, get_dataloader
from ..utils.training import count_nfe
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


def compare_regularizations():
    """Compara diferentes combinações de regularização."""
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
            n_epochs=15,
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
                [dataset[i][0] for i in range(100)]
            ).to(device)
            final_log_prob = model.log_prob(test_batch).mean().item()

        results[config_name] = {
            'lambda_ke': lambda_ke,
            'lambda_jf': lambda_jf,
            'nfe': nfe,
            'final_log_prob': final_log_prob,
            'history': history,
            'model': model
        }

        print(
            f"NFEs: {nfe}, Final log-prob: {final_log_prob:.4f}"
        )

    return results


def plot_convergence_analysis(
    results: Dict[str, Dict],
    save_dir: str = 'results/figures'
) -> None:
    """Plot training convergence curves for all regularization configs.

    Args:
        results: Dictionary with results from compare_regularizations().
        save_dir: Directory to save figures.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_loss, ax_nll = axes[0]
    ax_ke, ax_jf = axes[1]

    # Color map for different configs
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (config_name, result), color in zip(results.items(), colors):
        history = result['history']
        epochs = range(1, len(history['loss']) + 1)

        # Plot loss
        ax_loss.plot(
            epochs, history['loss'],
            label=config_name, color=color, linewidth=2, alpha=0.8
        )

        # Plot NLL
        ax_nll.plot(
            epochs, history['nll'],
            label=config_name, color=color, linewidth=2, alpha=0.8
        )

        # Plot KE
        ax_ke.plot(
            epochs, history['ke'],
            label=config_name, color=color, linewidth=2, alpha=0.8
        )

        # Plot JF
        ax_jf.plot(
            epochs, history['jf'],
            label=config_name, color=color, linewidth=2, alpha=0.8
        )

    # Format loss plot
    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Total Loss', fontsize=12)
    ax_loss.set_title(
        'Training Loss Convergence', fontsize=14, fontweight='bold'
    )
    ax_loss.legend(fontsize=8, loc='best')
    ax_loss.grid(True, alpha=0.3)

    # Format NLL plot
    ax_nll.set_xlabel('Epoch', fontsize=12)
    ax_nll.set_ylabel('Negative Log-Likelihood', fontsize=12)
    ax_nll.set_title('NLL Convergence', fontsize=14, fontweight='bold')
    ax_nll.legend(fontsize=8, loc='best')
    ax_nll.grid(True, alpha=0.3)

    # Format KE plot
    ax_ke.set_xlabel('Epoch', fontsize=12)
    ax_ke.set_ylabel('Kinetic Energy', fontsize=12)
    ax_ke.set_title(
        'Kinetic Energy Regularization', fontsize=14, fontweight='bold'
    )
    ax_ke.legend(fontsize=8, loc='best')
    ax_ke.grid(True, alpha=0.3)

    # Format JF plot
    ax_jf.set_xlabel('Epoch', fontsize=12)
    ax_jf.set_ylabel('Jacobian Frobenius Norm', fontsize=12)
    ax_jf.set_title(
        'Jacobian Frobenius Regularization', fontsize=14, fontweight='bold'
    )
    ax_jf.legend(fontsize=8, loc='best')
    ax_jf.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'exp2_convergence_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence analysis saved to: {save_path}")
    plt.close()


def plot_log_likelihood_comparison(
    results: Dict[str, Dict],
    save_dir: str = 'results/figures'
) -> None:
    """Compare final log-likelihoods across different regularization configs.

    Args:
        results: Dictionary with results from compare_regularizations().
        save_dir: Directory to save figures.
    """
    os.makedirs(save_dir, exist_ok=True)

    config_names = list(results.keys())
    log_probs = [results[name]['final_log_prob'] for name in config_names]
    nfes = [results[name]['nfe'] for name in config_names]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Log-likelihood comparison
    colors = plt.cm.viridis(np.linspace(0, 1, len(config_names)))
    bars1 = ax1.barh(config_names, log_probs, color=colors, alpha=0.7)
    ax1.set_xlabel('Final Log-Likelihood', fontsize=12)
    ax1.set_title('Log-Likelihood Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, log_probs)):
        ax1.text(
            val, i, f' {val:.4f}',
            va='center', fontsize=9, fontweight='bold'
        )

    # Plot 2: NFE comparison
    bars2 = ax2.barh(config_names, nfes, color=colors, alpha=0.7)
    ax2.set_xlabel(
        'Number of Function Evaluations (NFE)', fontsize=12
    )
    ax2.set_title('NFE Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, nfes)):
        ax2.text(
            val, i, f' {val}',
            va='center', fontsize=9, fontweight='bold'
        )

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'exp2_loglikelihood_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Log-likelihood comparison saved to: {save_path}")
    plt.close()


def plot_sample_quality_comparison(
    results: Dict[str, Dict],
    dataset: Synthetic2D,
    n_samples: int = 1000,
    save_dir: str = 'results/figures'
) -> None:
    """Visualize sample quality from different regularization configs.

    Args:
        results: Dictionary with results from compare_regularizations().
        dataset: Original dataset for reference.
        n_samples: Number of samples to generate.
        save_dir: Directory to save figures.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_configs = len(results)
    n_cols = 3
    n_rows = (n_configs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Plot original data in first subplot
    data_np = dataset.data.cpu().numpy()
    axes[0].scatter(
        data_np[:, 0], data_np[:, 1],
        alpha=0.5, s=10, c='blue'
    )
    axes[0].set_title('Original Data', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(r'$x_1$', fontsize=10)
    axes[0].set_ylabel(r'$x_2$', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')

    # Plot samples from each model
    for idx, (config_name, result) in enumerate(results.items(), start=1):
        model = result['model']
        model.eval()

        with torch.no_grad():
            samples = model.sample(n_samples)
            samples_np = samples.cpu().numpy()

        axes[idx].scatter(
            samples_np[:, 0], samples_np[:, 1],
            alpha=0.5, s=10, c='red'
        )
        title = (
            f'{config_name}\n'
            f'(LL: {result["final_log_prob"]:.3f}, NFE: {result["nfe"]})'
        )
        axes[idx].set_title(title, fontsize=10, fontweight='bold')
        axes[idx].set_xlabel(r'$x_1$', fontsize=10)
        axes[idx].set_ylabel(r'$x_2$', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axis('equal')

    # Hide unused subplots
    for idx in range(len(results) + 1, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(
        'Sample Quality Comparison Across Regularization Configs',
        fontsize=16, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'exp2_sample_quality_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sample quality comparison saved to: {save_path}")
    plt.close()


def plot_transformation_comparison(
    results: Dict[str, Dict],
    save_dir: str = 'results/figures'
) -> None:
    """Visualize transformations from z ~ N(0,I) to x for each config.

    Args:
        results: Dictionary with results from compare_regularizations().
        save_dir: Directory to save figures.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_configs = len(results)
    n_cols = 2
    n_rows = n_configs

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_samples = 500

    for idx, (config_name, result) in enumerate(results.items()):
        model = result['model']
        ax_left, ax_right = axes[idx]

        # Sample z ~ N(0, I)
        z = torch.randn(n_samples, 2, device=device)

        # Transform z -> x using FFJORD
        model.eval()
        with torch.no_grad():
            x, _ = model.backward(z)

        z_np = z.cpu().numpy()
        x_np = x.cpu().numpy()

        # LEFT PLOT: Initial z samples
        ax_left.scatter(
            z_np[:, 0], z_np[:, 1],
            marker='o', s=10, alpha=0.6, linewidths=1
        )
        from matplotlib.patches import Circle
        circle = Circle(
            (0, 0), 2, fill=False, linestyle='--', color='gray'
        )
        ax_left.add_patch(circle)
        ax_left.set_xlabel(r'$x_1$', fontsize=10)
        ax_left.set_ylabel(r'$x_2$', fontsize=10)
        ax_left.set_title(
            f'{config_name} - Base Distribution',
            fontsize=11, fontweight='bold'
        )
        ax_left.set_xlim(-3, 3)
        ax_left.set_ylim(-3, 3)
        ax_left.grid(True, alpha=0.3)
        ax_left.axis('equal')

        # RIGHT PLOT: Transformed samples
        ax_right.scatter(
            x_np[:, 0], x_np[:, 1],
            alpha=0.5, s=10
        )
        ax_right.set_xlabel(r'$x_1$', fontsize=10)
        ax_right.set_ylabel(r'$x_2$', fontsize=10)
        ax_right.set_title(
            f'{config_name} - Transformed Samples',
            fontsize=11, fontweight='bold'
        )
        ax_right.grid(True, alpha=0.3)
        ax_right.axis('equal')

    plt.suptitle(
        'Transformation Comparison: z ~ N(0,I) → x',
        fontsize=16, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'exp2_transformation_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Transformation comparison saved to: {save_path}")
    plt.close()


def analyze_regularization_impact(
    results: Dict[str, Dict],
    dataset: Synthetic2D,
    save_dir: str = 'results/figures'
) -> None:
    """Comprehensive analysis of regularization impact.

    Creates all analysis plots and saves summary statistics.

    Args:
        results: Dictionary with results from compare_regularizations().
        dataset: Original dataset for reference.
        save_dir: Directory to save figures and summary.
    """
    print("\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    # 1. Convergence analysis
    print("\n1. Plotting convergence curves...")
    plot_convergence_analysis(results, save_dir)

    # 2. Log-likelihood comparison
    print("\n2. Comparing log-likelihoods...")
    plot_log_likelihood_comparison(results, save_dir)

    # 3. Sample quality comparison
    print("\n3. Visualizing sample quality...")
    plot_sample_quality_comparison(results, dataset, save_dir=save_dir)

    # 4. Transformation comparison
    print("\n4. Comparing transformations...")
    plot_transformation_comparison(results, save_dir)

    # 5. Summary statistics
    print("\n5. Generating summary statistics...")
    summary_path = os.path.join(save_dir, 'exp2_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("REGULARIZATION IMPACT ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("Configuration Comparison:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Config':<30} {'Log-Prob':<15} {'NFE':<10}\n")
        f.write("-" * 60 + "\n")

        for config_name, result in results.items():
            f.write(
                f"{config_name:<30} "
                f"{result['final_log_prob']:<15.4f} "
                f"{result['nfe']:<10}\n"
            )

        f.write("\n" + "=" * 60 + "\n")
        f.write("KEY INSIGHTS:\n")
        f.write("=" * 60 + "\n\n")

        # Find best log-likelihood
        best_ll = max(results.items(), key=lambda x: x[1]['final_log_prob'])
        f.write(
            f"Best Log-Likelihood: {best_ll[0]} "
            f"({best_ll[1]['final_log_prob']:.4f})\n"
        )

        # Find lowest NFE
        lowest_nfe = min(results.items(), key=lambda x: x[1]['nfe'])
        f.write(f"Lowest NFE: {lowest_nfe[0]} ({lowest_nfe[1]['nfe']})\n")

        # Compare baseline vs regularized
        baseline = results.get('λ_KE=0.0, λ_JF=0.0')
        if baseline:
            f.write("\nBaseline (no regularization):\n")
            f.write(f"  Log-Prob: {baseline['final_log_prob']:.4f}\n")
            f.write(f"  NFE: {baseline['nfe']}\n")

            for config_name, result in results.items():
                if config_name != 'λ_KE=0.0, λ_JF=0.0':
                    ll_diff = (
                        result['final_log_prob'] -
                        baseline['final_log_prob']
                    )
                    nfe_diff = result['nfe'] - baseline['nfe']
                    f.write(f"\n{config_name}:\n")
                    f.write(f"  Log-Prob diff: {ll_diff:+.4f}\n")
                    f.write(f"  NFE diff: {nfe_diff:+d}\n")

    print(f"Summary saved to: {summary_path}")
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
