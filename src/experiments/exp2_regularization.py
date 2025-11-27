"""Experimento 2: Regularização e tolerâncias."""
import torch
import torch.optim as optim
from ..models.cnf import CNF
from ..models.vector_field import VectorField
from ..utils.datasets import Synthetic2D, get_dataloader
from ..utils.training import count_nfe
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict


def compute_regularizations(
    vf: VectorField, x: torch.Tensor, t: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Computa termos de regularização.

    Args:
        vf: Vector field module.
        x: Input tensor with shape (batch, features).
        t: Time tensor (scalar or batch).

    Returns:
        Dictionary with 'kinetic_energy' and 'jacobian_frobenius' keys.
    """
    x = x.requires_grad_(True)
    v = vf(t, x)  # (batch, d)

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


def train_cnf_with_regularization(
    model: CNF,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    lambda_ke: float = 0.0,
    lambda_jf: float = 0.0,
    reg_time: float = 0.5
) -> Dict[str, list]:
    """Train CNF with regularization terms.

    Args:
        model: CNF model.
        dataloader: DataLoader for training data.
        optimizer: Optimizer for training.
        device: Device to run training on.
        num_epochs: Number of training epochs.
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

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_nll = 0.0
        total_ke = 0.0
        total_jf = 0.0
        n_batches = 0

        for batch in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            x = batch[0].to(device)
            optimizer.zero_grad()

            # Calculate log-likelihood
            log_prob = model.log_prob(x)
            nll = -log_prob.mean()

            # Compute regularizations at time reg_time
            t_reg = torch.tensor(reg_time, device=device)
            regs = compute_regularizations(model.vf, x, t_reg)
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
        vf = VectorField(
            features=2, hidden_dims=[64, 64], time_embed_dim=16
        )
        model = CNF(vf, method='dopri5', rtol=1e-3, atol=1e-4).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Treinar com regularização
        history = train_cnf_with_regularization(
            model,
            dataloader,
            optimizer,
            device,
            num_epochs=50,
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


if __name__ == '__main__':
    results = compare_regularizations()
    print("\n=== Resumo ===")
    for config_name, result in results.items():
        print(
            f"{config_name}: "
            f"NFEs={result['nfe']}, "
            f"log-prob={result['final_log_prob']:.4f}"
        )
