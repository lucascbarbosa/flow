"""Experimento 1: Comparação de solvers ODE."""
import os
import csv
import time
import torch
import torch.optim as optim
from src.models.neural_ode import NeuralODE
from src.models.vector_field import VectorField2D
from src.utils.datasets import Synthetic2D, get_dataloader
from src.utils.training import train_neural_ode, count_nfe
from src.utils.visualization import Synthetic2DViz


solvers_config = [
    {'method': 'euler', 'rtol': None, 'atol': None},  # Fixed step
    {'method': 'rk4', 'rtol': None, 'atol': None},
    {'method': 'dopri5', 'rtol': 1e-3, 'atol': 1e-4},
    {'method': 'dopri5', 'rtol': 1e-4, 'atol': 1e-5},
    {'method': 'dopri5', 'rtol': 1e-5, 'atol': 1e-6},
]

dataset_types = ['moons', 'circles', 'spirals']


def compare_solvers(
    checkpoint_dir: str = 'results/checkpoints/exp1',
    resume: bool = True,
    n_epochs: int = 50
):
    """Compara diferentes solvers ODE.

    Args:
        checkpoint_dir: Directory to save/load checkpoints.
        resume: If True, load existing checkpoints instead of retraining.
        n_epochs: Number of training epochs if training from scratch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(checkpoint_dir, exist_ok=True)

    results = {}
    all_metrics = []

    for dataset_type in dataset_types:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_type}")
        print(f"{'=' * 60}")

        # Dataset
        dataset = Synthetic2D(
            n_samples=5000,
            noise=0.05,
            dataset_type=dataset_type
        )
        dataloader = get_dataloader(dataset, batch_size=128, shuffle=True)

        results[dataset_type] = {}

        for config in solvers_config:
            method = config['method']
            rtol = config['rtol']
            atol = config['atol']

            # Create a descriptive key for results
            if rtol is None and atol is None:
                solver_key = method
            else:
                solver_key = f"{method}_rtol{rtol}_atol{atol}"

            print(f"\n=== Testando solver: {solver_key} ===")

            # Checkpoint path
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'exp1_{dataset_type}_{solver_key}.pt'
            )

            # Load checkpoint if exists, otherwise train
            if resume and os.path.exists(checkpoint_path):
                print(f"Carregando checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)

                # Reconstruct model
                model_config = checkpoint['model_config']
                vf = VectorField2D(
                    features=model_config['features'],
                    hidden_dims=model_config['hidden_dims'],
                    time_embed_dim=model_config['time_embed_dim']
                )
                model = NeuralODE(
                    vf,
                    solver=model_config.get('solver', method),
                    rtol=model_config.get('rtol', rtol),
                    atol=model_config.get('atol', atol)
                ).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])

                # Load metrics
                final_loss = checkpoint.get('final_loss', None)
                nfe = checkpoint.get('nfe', None)

                # Compute reconstruction error as quality metric
                model.eval()
                with torch.no_grad():
                    test_samples = torch.stack(
                        [dataset[i] for i in range(min(50, len(dataset)))]
                    ).to(device)
                    # Forward: x -> z
                    forward_traj = model(test_samples, n_steps=100)
                    z = forward_traj[-1]  # Final state
                    # Backward: z -> x_recon
                    backward_traj = model.backward(z, n_steps=100)
                    x_recon = backward_traj[-1]  # Final state
                    # Reconstruction error
                    recon_error = ((test_samples - x_recon) ** 2).mean().item()

                print(
                    f"Loaded: Loss={final_loss}, NFEs={nfe}, "
                    f"Recon Error={recon_error:.6f}"
                )
            else:
                print("Treinando modelo (checkpoint não encontrado)...")

                # Modelo
                vf = VectorField2D(
                    features=2,
                    hidden_dims=[64, 64],
                    time_embed_dim=16
                )
                model = NeuralODE(
                    vf,
                    solver=method,
                    rtol=rtol,
                    atol=atol
                ).to(device)
                optimizer = optim.Adam(model.parameters(), lr=1e-4)

                # Train
                final_loss = train_neural_ode(
                    model,
                    dataloader,
                    optimizer,
                    n_epochs=n_epochs
                )

                # Contar NFEs
                sample_batch = torch.randn(100, 2).to(device)
                nfe = count_nfe(model, sample_batch)

                # Save checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'model_config': {
                        'features': 2,
                        'hidden_dims': [64, 64],
                        'time_embed_dim': 16,
                        'solver': method,
                        'rtol': rtol,
                        'atol': atol
                    },
                    'final_loss': final_loss,
                    'nfe': nfe,
                    'dataset_type': dataset_type,
                    'solver_key': solver_key
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint salvo em: {checkpoint_path}")

            # Medir tempo de forward pass
            model.eval()
            with torch.no_grad():
                sample_batch = torch.randn(100, 2).to(device)
                start_time = time.time()
                _ = model(sample_batch, n_steps=100)
                forward_time = time.time() - start_time

            # Medir tempo de backward pass
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                _ = model.backward(sample_batch, n_steps=100)
                backward_time = time.time() - start_time

            # Compute reconstruction error as quality metric
            # (forward then backward should return to original)
            model.eval()
            with torch.no_grad():
                test_samples = torch.stack(
                    [dataset[i] for i in range(min(50, len(dataset)))]
                ).to(device)
                # Forward: x -> z
                forward_traj = model(test_samples, n_steps=100)
                z = forward_traj[-1]  # Final state
                # Backward: z -> x_recon
                backward_traj = model.backward(z, n_steps=100)
                x_recon = backward_traj[-1]  # Final state
                # Reconstruction error
                recon_error = ((test_samples - x_recon) ** 2).mean().item()
            # Note: Log-likelihood is not applicable to NeuralODE
            # (trained with MMD loss). Using reconstruction error as a
            # quality metric instead.

            # Compute reconstruction error if not already computed
            # (from checkpoint)
            if 'recon_error' not in locals():
                model.eval()
                with torch.no_grad():
                    test_samples = torch.stack(
                        [dataset[i] for i in range(min(50, len(dataset)))]
                    ).to(device)
                    # Forward: x -> z
                    forward_traj = model(test_samples, n_steps=100)
                    z = forward_traj[-1]  # Final state
                    # Backward: z -> x_recon
                    backward_traj = model.backward(z, n_steps=100)
                    x_recon = backward_traj[-1]  # Final state
                    # Reconstruction error
                    recon_error = ((test_samples - x_recon) ** 2).mean().item()

            results[dataset_type][solver_key] = {
                'nfe': nfe,
                'loss': final_loss,
                'model': model,
                'forward_time': forward_time,
                'backward_time': backward_time,
                'method': method,
                'rtol': rtol,
                'atol': atol,
                'recon_error': recon_error
            }

            # Store metrics for CSV
            all_metrics.append({
                'Dataset': dataset_type,
                'Config': solver_key,
                'Loss': final_loss,
                'NFE': nfe,
                'Forward_Time': forward_time,
                'Backward_Time': backward_time,
                'Method': method,
                'RTOL': rtol if rtol is not None else '',
                'ATOL': atol if atol is not None else '',
                'Recon_Error': recon_error,
                'Log_Prob': ''  # Not applicable to NeuralODE
            })

            print(f"NFEs: {nfe}")
            print(f"Loss: {final_loss:.4f}")
            print(f"Tempo forward: {forward_time:.4f}s")
            print(f"Tempo backward: {backward_time:.4f}s")

    # Save CSV
    csv_path = os.path.join('results', 'exp1_metrics.csv')
    os.makedirs('results', exist_ok=True)
    if all_metrics:
        fieldnames = [
            'Dataset', 'Config', 'Loss', 'NFE',
            'Forward_Time', 'Backward_Time', 'Method', 'RTOL', 'ATOL',
            'Recon_Error', 'Log_Prob'
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\nCSV metrics saved to: {csv_path}")

    # Generate plots for each dataset type
    print("\n" + "=" * 60)
    print("GENERATING TRANSFORMATION PLOTS")
    print("=" * 60)
    for dataset_type in dataset_types:
        print(f"\nGenerating plots for dataset: {dataset_type}")
        models_list = []
        save_paths = []

        for solver_key in results[dataset_type].keys():
            model = results[dataset_type][solver_key]['model']
            models_list.append(model)

            # Create save path
            plot_dir = os.path.join(
                'results', 'figures', 'exp1', dataset_type
            )
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(
                plot_dir, f'transformation_{solver_key}.png'
            )
            save_paths.append(plot_path)

        # Plot transformations for all solvers of this dataset
        Synthetic2DViz.plot_transformation(
            models_list,
            n_samples=1000,
            n_steps=100,
            save_path=save_paths
        )
        print(f"Plots saved for {dataset_type}")

    return results


if __name__ == '__main__':
    results = compare_solvers(n_epochs=100)
    print("\n=== Resumo ===")
    for dataset_type, dataset_results in results.items():
        print(f"\nDataset: {dataset_type}")
        for solver, result in dataset_results.items():
            print(
                f"  {solver}: "
                f"Loss={result['loss']:.4f}, "
                f"NFEs={result['nfe']}, "
                f"forward: {result['forward_time']:.4f}s, "
                f"backward: {result['backward_time']:.4f}s"
            )
