"""Experimento 1: Comparação de solvers ODE."""
import os
import time
import torch
import torch.optim as optim
from src.models.neural_ode import NeuralODE
from src.models.vector_field import VectorField2D
from src.utils.datasets import Synthetic2D, get_dataloader
from src.utils.training import train_neural_ode, count_nfe


solvers_config = [
    {'method': 'euler', 'rtol': None, 'atol': None},
    {'method': 'rk4', 'rtol': None, 'atol': None},
    {'method': 'dopri5', 'rtol': 1e-3, 'atol': 1e-4},
    {'method': 'dopri5', 'rtol': 1e-4, 'atol': 1e-5},
    {'method': 'dopri5', 'rtol': 1e-5, 'atol': 1e-6},
]

dataset_types = ['moons', 'circles', 'spirals']


def compare_solvers():
    """Compara diferentes solvers ODE."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Checkpoint directory
    checkpoint_dir = os.path.join('results', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    results = {}

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
                atol=atol,
            ).to(device)

            # Checkpoint path using existing naming pattern
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'01__{dataset_type}_neuralode.pt'
            )

            # Load checkpoint if exists, otherwise train
            if os.path.exists(checkpoint_path):
                print(f"Carregando checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint)
                loss = None  # Loss not stored in checkpoint
            else:
                print("Treinando modelo (checkpoint não encontrado)...")
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                loss = train_neural_ode(
                    model,
                    dataloader,
                    optimizer,
                    n_epochs=50
                )
                # Save checkpoint
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint salvo em: {checkpoint_path}")

            # Contar NFEs (usando amostras de N(0, I))
            sample_batch = torch.randn(100, 2).to(device)
            nfe = count_nfe(model, sample_batch)

            # Medir tempo de forward pass
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                _ = model(sample_batch, n_steps=100)
                forward_time = time.time() - start_time

            # Medir tempo de backward pass
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                _ = model.backward(sample_batch, n_steps=100)
                backward_time = time.time() - start_time

            results[dataset_type][solver_key] = {
                'nfe': nfe,
                'loss': loss,
                'model': model,
                'forward_time': forward_time,
                'backward_time': backward_time,
            }

            print(f"NFEs: {nfe}")
            print(f"Tempo forward: {forward_time:.4f}s")
            print(f"Tempo backward: {backward_time:.4f}s")

    return results


if __name__ == '__main__':
    results = compare_solvers()
    print("\n=== Resumo ===")
    for dataset_type, dataset_results in results.items():
        print(f"\nDataset: {dataset_type}")
        for solver, result in dataset_results.items():
            print(
                f"  {solver}: "
                f"{result['nfe']} NFEs, "
                f"forward: {result['forward_time']:.4f}s, "
                f"backward: {result['backward_time']:.4f}s"
            )
