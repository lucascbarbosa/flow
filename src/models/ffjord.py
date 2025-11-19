"""
FFJORD: CNF com Hutchinson trace estimator para escalabilidade.
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

from ..utils.trace import divergence_hutchinson


class FFJORD(nn.Module):
    """
    FFJORD: Continuous Normalizing Flow com Hutchinson trace estimator.
    Escalável para alta dimensão!
    """
    def __init__(self, vector_field, base_dist=None, num_samples=1, distribution='rademacher'):
        super().__init__()
        self.vf = vector_field
        self.num_samples = num_samples
        self.distribution = distribution
        
        if base_dist is None:
            # Prior: N(0, I)
            features = vector_field.features
            self.base_dist = torch.distributions.MultivariateNormal(
                torch.zeros(features),
                torch.eye(features)
            )
        else:
            self.base_dist = base_dist
    
    def _augmented_dynamics(self, t, state):
        """
        Augmented ODE: integra [x, log_det] simultaneamente.
        dx/dt = f(x, t)
        d(log_det)/dt = -trace(∂f/∂x) [estimado via Hutchinson]
        
        Args:
            t: tempo escalar
            state: (batch, features + 1) # [x, log_det]
        Returns:
            d_state: (batch, features + 1)
        """
        batch_size = state.shape[0]
        x = state[:, :-1]  # (batch, features)
        log_det = state[:, -1:]  # (batch, 1)
        
        # Habilitar gradientes para x
        x = x.requires_grad_(True)
        
        # Compute vector field
        dx_dt = self.vf(t, x)  # (batch, features)
        
        # Compute trace do Jacobiano usando Hutchinson estimator
        trace = divergence_hutchinson(
            lambda x: self.vf(t, x),
            x,
            num_samples=self.num_samples,
            distribution=self.distribution
        )  # (batch,)
        
        # d(log_det)/dt = -trace (note o sinal!)
        dlogdet_dt = -trace.unsqueeze(-1)  # (batch, 1)
        
        return torch.cat([dx_dt, dlogdet_dt], dim=-1)
    
    def forward(self, x, reverse=False):
        """
        Transforma x -> z (forward) ou z -> x (reverse).
        
        Args:
            x: input (batch, features)
            reverse: se True, integra de t=1 para t=0 (z -> x)
        Returns:
            z: transformed (batch, features)
            log_det: log determinant (batch,)
        """
        if reverse:
            # z -> x: integra de t=1 para t=0
            t_span = torch.tensor([1., 0.], device=x.device, dtype=x.dtype)
        else:
            # x -> z: integra de t=0 para t=1
            t_span = torch.tensor([0., 1.], device=x.device, dtype=x.dtype)
        
        # Estado inicial: [x, log_det=0]
        log_det_init = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        state_init = torch.cat([x, log_det_init], dim=-1)
        
        # Integrar ODE aumentada
        state_t = odeint_adjoint(
            self._augmented_dynamics,
            state_init,
            t_span,
            method='dopri5',
            rtol=1e-3,
            atol=1e-4
        )
        
        # Estado final
        state_final = state_t[-1]  # (batch, features + 1)
        z = state_final[:, :-1]  # (batch, features)
        log_det = state_final[:, -1]  # (batch,)
        
        return z, log_det
    
    def log_prob(self, x):
        """
        Calcula log p(x) usando change of variables.
        
        Args:
            x: input (batch, features)
        Returns:
            log_prob: (batch,)
        """
        # Transformar x -> z
        z, log_det = self.forward(x, reverse=False)
        
        # log p(x) = log p(z) + log |det(∂z/∂x)|
        log_prob_z = self.base_dist.log_prob(z)
        log_prob_x = log_prob_z + log_det
        
        return log_prob_x
    
    def sample(self, num_samples):
        """
        Gera amostras x ~ p(x) via z ~ p(z) -> x.
        
        Args:
            num_samples: número de amostras
        Returns:
            x: samples (num_samples, features)
        """
        # Sample z ~ p(z)
        z = self.base_dist.sample((num_samples,))
        
        # Transformar z -> x
        x, _ = self.forward(z, reverse=True)
        
        return x

