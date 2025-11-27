"""Experimento 2: Regularização e tolerâncias."""
import torch
import torch.optim as optim
from ..models.neural_ode import NeuralODE
from ..models.vector_field import VectorField
from ..utils.datasets import Synthetic2D, get_dataloader
from ..utils.training import train_neural_ode, count_nfe


def compare_tolerances():
    """Compara diferentes tolerâncias rtol/atol."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    dataset = Synthetic2D(n_samples=5000, noise=0.05, dataset_type='moons')
    dataloader = get_dataloader(dataset, batch_size=128, shuffle=True)

    # Tolerâncias para comparar
    tolerances = [
        (1e-2, 1e-3),
        (1e-3, 1e-4),
        (1e-4, 1e-5),
    ]
    results = {}

    for rtol, atol in tolerances:
        print(f"\n=== Testando rtol={rtol}, atol={atol} ===")

        # Modelo
        vf = VectorField(features=2, hidden_dims=[64, 64], time_embed_dim=16)
        model = NeuralODE(vf, solver='dopri5', rtol=rtol, atol=atol).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Treinar
        train_neural_ode(model, dataloader, optimizer, device, num_epochs=10)

        # Contar NFEs (usando amostras de N(0, I))
        sample_batch = torch.randn(10, 2).to(device)
        nfe = count_nfe(model, sample_batch)

        results[(rtol, atol)] = {
            'nfe': nfe,
            'model': model
        }

        print(f"NFEs: {nfe}")

    return results


if __name__ == '__main__':
    results = compare_tolerances()
    print("\n=== Resumo ===")
    for tol, result in results.items():
        print(f"rtol={tol[0]}, atol={tol[1]}: {result['nfe']} NFEs")
