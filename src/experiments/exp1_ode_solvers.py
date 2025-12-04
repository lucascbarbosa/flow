"""Experimento 1: Comparação de solvers ODE."""
import torch
import torch.optim as optim
from src.models.neural_ode import NeuralODE
from src.models.vector_field import VectorField
from src.utils.datasets import Synthetic2D, get_dataloader
from src.utils.training import train_neural_ode, count_nfe


solvers_config = [
    {'method': 'euler', 'rtol': None, 'atol': None},
    {'method': 'rk4', 'rtol': None, 'atol': None},
    {'method': 'dopri5', 'rtol': 1e-3, 'atol': 1e-4},
    {'method': 'dopri5', 'rtol': 1e-4, 'atol': 1e-5},
    {'method': 'dopri5', 'rtol': 1e-5, 'atol': 1e-6},
]


def compare_solvers():
    """Compara diferentes solvers ODE."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = Synthetic2D(n_samples=5000, noise=0.05, dataset_type='moons')
    dataloader = get_dataloader(dataset, batch_size=128, shuffle=True)

    results = {}

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

        # Modelo (moons dataset has 2 classes)
        vf = VectorField(features=2, hidden_dims=[64, 64], time_embed_dim=16)
        model = NeuralODE(
            vf,
            solver=method,
            rtol=rtol,
            atol=atol,
            n_outputs=2
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Treinar
        loss = train_neural_ode(
            model,
            dataloader,
            optimizer,
            device,
            num_epochs=50
        )

        # Contar NFEs (usando amostras de N(0, I))
        sample_batch = torch.randn(100, 2).to(device)
        nfe = count_nfe(model, sample_batch)

        results[solver_key] = {'nfe': nfe, 'loss': loss, 'model': model}

        print(f"NFEs: {nfe}")

    return results


if __name__ == '__main__':
    results = compare_solvers()
    print("\n=== Resumo ===")
    for solver, result in results.items():
        print(f"{solver}: {result['nfe']} NFEs")
