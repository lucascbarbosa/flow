"""
Experimento 3: Comparação de arquiteturas de Vector Field.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.vector_field import VectorField, ResNetVF, TimeConditionedVF
from ..models.neural_ode import NeuralODE
from ..utils.datasets import Synthetic2D, get_dataloader
from ..utils.training import train_neural_ode


def compare_architectures():
    """Compara diferentes arquiteturas de Vector Field."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    dataset = Synthetic2D(n_samples=5000, noise=0.05, dataset_type='moons')
    dataloader = get_dataloader(dataset, batch_size=128, shuffle=True)
    
    # Arquiteturas para comparar
    architectures = {
        'SimpleMLP': VectorField,
        'ResNetVF': ResNetVF,
        'TimeConditionedVF': TimeConditionedVF,
    }
    results = {}
    
    for name, vf_class in architectures.items():
        print(f"\n=== Testando arquitetura: {name} ===")
        
        # Modelo
        vf = vf_class(features=2, hidden_dims=[64, 64], time_embed_dim=16)
        model = NeuralODE(vf, solver='dopri5', rtol=1e-3, atol=1e-4).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Treinar
        train_neural_ode(model, dataloader, optimizer, device, num_epochs=10)
        
        results[name] = {
            'model': model
        }
    
    return results


if __name__ == '__main__':
    results = compare_architectures()
    print("\n=== Resumo ===")
    for arch, result in results.items():
        print(f"{arch}: Treinado")

