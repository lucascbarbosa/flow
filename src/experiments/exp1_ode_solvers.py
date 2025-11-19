"""
Experimento 1: Comparação de solvers ODE.
"""
import torch
import torch.optim as optim

from ..models.vector_field import VectorField
from ..models.neural_ode import NeuralODE
from ..utils.datasets import Synthetic2D, get_dataloader
from ..utils.training import train_neural_ode, count_nfe


def compare_solvers():
    """Compara diferentes solvers ODE."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = Synthetic2D(n_samples=5000, noise=0.05, dataset_type='moons')
    dataloader = get_dataloader(dataset, batch_size=128, shuffle=True)

    # Solvers para comparar
    solvers = ['euler', 'rk4', 'dopri5']
    results = {}

    for solver in solvers:
        print(f"\n=== Testando solver: {solver} ===")

        # Modelo
        vf = VectorField(features=2, hidden_dims=[64, 64], time_embed_dim=16)
        model = NeuralODE(vf, solver=solver, rtol=1e-3, atol=1e-4).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Treinar
        train_neural_ode(model, dataloader, optimizer, device, num_epochs=10)

        # Contar NFEs (usando amostras de N(0, I))
        sample_batch = torch.randn(10, 2).to(device)
        nfe = count_nfe(model, sample_batch)

        results[solver] = {'nfe': nfe, 'model': model}

        print(f"NFEs: {nfe}")

    return results


if __name__ == '__main__':
    results = compare_solvers()
    print("\n=== Resumo ===")
    for solver, result in results.items():
        print(f"{solver}: {result['nfe']} NFEs")
