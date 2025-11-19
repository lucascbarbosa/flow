"""
Trace computation utilities for CNF.
"""
import torch
from torch import autograd


def divergence_exact(f, x):
    """
    Calcula trace(∂f/∂x) exatamente usando autograd.
    ATENÇÃO: Custo O(d²) - só viável para dimensão baixa!
    
    Args:
        f: função R^d -> R^d (callable que aceita x e retorna (batch, d))
        x: input (batch, d)
    Returns:
        trace: (batch,)
    """
    batch_size, dim = x.shape
    
    # Habilitar gradientes
    x = x.requires_grad_(True)
    
    # Compute f(x)
    f_x = f(x)  # (batch, dim)
    
    # Para cada dimensão i, calcular ∂f_i/∂x_i
    trace = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
    
    for i in range(dim):
        # Compute ∂f[:, i]/∂x
        df_i = autograd.grad(
            f_x[:, i].sum(),
            x,
            create_graph=True,
            retain_graph=True
        )[0]  # (batch, dim)
        
        # Somar apenas a diagonal: ∂f_i/∂x_i
        trace += df_i[:, i]
    
    return trace


def divergence_hutchinson(f, x, num_samples=1, distribution='rademacher'):
    """
    Calcula trace(∂f/∂x) usando Hutchinson estimator.
    Custo O(d) - escalável para alta dimensão!
    
    Args:
        f: função R^d -> R^d (callable que aceita x e retorna (batch, d))
        x: input (batch, d)
        num_samples: número de amostras para estimativa (default: 1)
        distribution: 'rademacher' ou 'gaussian'
    Returns:
        trace: (batch,)
    """
    batch_size, dim = x.shape
    
    # Habilitar gradientes
    x = x.requires_grad_(True)
    
    # Compute f(x)
    f_x = f(x)  # (batch, dim)
    
    trace_estimates = []
    
    for _ in range(num_samples):
        # Sample noise vector
        if distribution == 'rademacher':
            # Rademacher: ε ~ Uniform({-1, +1})
            epsilon = torch.randint(0, 2, (batch_size, dim), device=x.device, dtype=x.dtype) * 2 - 1
        elif distribution == 'gaussian':
            # Gaussian: ε ~ N(0, I)
            epsilon = torch.randn(batch_size, dim, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Compute ε^T * f(x)
        vTf = (epsilon * f_x).sum(dim=-1)  # (batch,)
        
        # Compute gradient: ∂(ε^T * f(x))/∂x = ε^T * ∂f/∂x
        grad_vTf = autograd.grad(
            vTf.sum(),
            x,
            create_graph=True,
            retain_graph=True
        )[0]  # (batch, dim)
        
        # Hutchinson estimator: ε^T * (∂f/∂x) * ε
        trace_est = (epsilon * grad_vTf).sum(dim=-1)  # (batch,)
        trace_estimates.append(trace_est)
    
    # Média sobre amostras
    trace = torch.stack(trace_estimates, dim=0).mean(dim=0)  # (batch,)
    
    return trace

