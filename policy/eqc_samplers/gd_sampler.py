import torch
from .base_sampler import BaseSampler


class GDSampler(BaseSampler):
    """Vanilla gradient descent on the stationary velocity field.

    x_{t+1} = x_t + eta * v(x_t)
    """

    def __init__(self):
        super().__init__()

    def step(self, stepsize, trajectory, model, global_cond=None):
        # Compute ∇_x (x · f_θ(x)) instead of raw f_θ(x).
        with torch.enable_grad():
            xt = trajectory.detach().requires_grad_(True)
            f = model(xt, torch.zeros(xt.shape[0], device=xt.device), global_cond)
            energy = torch.sum(xt * f)
            grad_energy = torch.autograd.grad(energy, xt, create_graph=False)[0]
        update = grad_energy * stepsize
        return update, grad_energy
