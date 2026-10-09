"""No-consistency EqC policy."""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import cleandiffuser.env.libero  # Registers the LIBERO Gym environments.
from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.diffusion.basic import DiffusionModel
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import TensorDict

from policy.eqc import create_transport
from policy.eqc_samplers import create_sampler
from policy.eqc_step_schedulers import create_step_scheduler
from policy.eqc_stoppers import create_stopper


class EqCModel(DiffusionModel):
    """Stationary energy-field policy trained only with gradient matching."""

    def __init__(
        self,
        nn_diffusion: BaseNNDiffusion,
        nn_condition: Optional[BaseNNCondition] = None,
        fix_mask: Optional[torch.Tensor] = None,
        loss_weight: Optional[torch.Tensor] = None,
        classifier: Optional[BaseClassifier] = None,
        ema_rate: float = 0.995,
        optimizer_params: Optional[dict] = None,
        x_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
        *,
        path_type: str = "Linear",
        prediction: str = "velocity",
        eqc_loss_weight: Optional[Any] = None,
        train_eps: Optional[float] = None,
        sample_eps: Optional[float] = None,
        per_action_noise: bool = False,
    ) -> None:
        super().__init__(nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, ema_rate, optimizer_params)
        if classifier is not None:
            raise ValueError("EqC does not support classifier guidance")
        self.x_max = nn.Parameter(x_max, requires_grad=False) if x_max is not None else None
        self.x_min = nn.Parameter(x_min, requires_grad=False) if x_min is not None else None
        self.transport = create_transport(
            path_type=path_type,
            prediction=prediction,
            loss_weight=eqc_loss_weight,
            train_eps=train_eps,
            sample_eps=sample_eps,
        )
        self.per_action_noise = per_action_noise

    @property
    def clip_pred(self) -> bool:
        return self.x_max is not None or self.x_min is not None

    def loss(self, x0: torch.Tensor, condition: Optional[Union[torch.Tensor, Dict]] = None) -> torch.Tensor:
        condition = self.model["condition"](condition) if condition is not None else None
        if self.per_action_noise:
            original_shape = x0.shape
            t, noise, data = self.transport.sample(x0.reshape(-1, 1))
            t, xt, target = self.transport.path_sampler.plan(t, noise, data)
            xt, target = xt.view(original_shape), (target * self.transport.get_ct(t)).view(original_shape)
        else:
            t, noise, data = self.transport.sample(x0)
            t, xt, target = self.transport.path_sampler.plan(t, noise, data)
            target = target * self.transport.get_ct(t)[:, None, None]

        xt = xt.detach().requires_grad_(True)
        velocity = self.model["diffusion"](xt, torch.zeros(xt.shape[0], device=xt.device), condition)
        energy_gradient = torch.autograd.grad(torch.sum(xt * velocity), xt, create_graph=True)[0]
        return ((energy_gradient - target).square() * (1 - self.fix_mask) * self.loss_weight).mean()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if "x0" not in batch:
            raise KeyError("Training batch must contain x0")
        loss = self.loss(batch["x0"], batch.get("condition_cfg"))
        self.log("train_loss", loss, prog_bar=True)
        if self.ema_update_schedule(batch_idx):
            self.ema_update()
        return loss

    def sample(
        self,
        prior: torch.Tensor,
        x1: Optional[torch.Tensor] = None,
        sampler_config: Optional[dict] = None,
        stopper_config: Optional[dict] = None,
        scheduler_config: Optional[dict] = None,
        use_ema: bool = True,
        temperature: float = 1.0,
        condition_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        mask_cfg: Optional[Union[torch.Tensor, TensorDict]] = None,
        num_candidates: int = 1,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        sampler = create_sampler(sampler_config)
        stopper = create_stopper(stopper_config)
        scheduler = create_step_scheduler(scheduler_config, default_stepsize=getattr(sampler, "stepsize", 1e-3))
        model = self.model_ema if use_ema else self.model
        prior = prior.to(self.device)
        with torch.no_grad():
            condition = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None

        if num_candidates > 1:
            prior = prior.repeat_interleave(num_candidates, dim=0)
            if x1 is not None:
                x1 = x1.repeat_interleave(num_candidates, dim=0)
            if condition is not None:
                condition = {key: value.repeat_interleave(num_candidates, dim=0) if isinstance(value, torch.Tensor) else value
                             for key, value in condition.items()}
        if x1 is None:
            x1 = torch.randn_like(prior) * temperature
        if prior.shape != x1.shape:
            raise ValueError("prior and x1 must have the same shape")

        trajectory = x1 * (1 - self.fix_mask) + prior * self.fix_mask
        sampler.reset_state(trajectory)
        stopper.reset_state(trajectory)
        active = torch.ones(trajectory.shape[0], dtype=torch.bool, device=self.device)
        steps = torch.zeros(trajectory.shape[0], device=self.device)
        for step_size in scheduler.get_schedule():
            active_before = active.clone()
            update, gradient = sampler.step(step_size, trajectory, model["diffusion"], condition)
            trajectory[active_before] = trajectory[active_before] + update[active_before]
            steps[active_before] += 1
            active = stopper.check(update=gradient, trajectory=trajectory, active_mask=active)
            trajectory = trajectory * (1 - self.fix_mask) + prior * self.fix_mask
            if not active.any():
                break

        if num_candidates > 1:
            with torch.no_grad():
                velocity = model["diffusion"](trajectory, torch.zeros(trajectory.shape[0], device=trajectory.device), condition)
                energies = (trajectory * velocity * (1 - self.fix_mask)).flatten(1).sum(1)
            original_batch = energies.shape[0] // num_candidates
            best = energies.view(original_batch, num_candidates).argmax(dim=1)
            trajectory = trajectory.view(original_batch, num_candidates, *trajectory.shape[1:])[
                torch.arange(original_batch, device=trajectory.device), best
            ]
            steps = steps.view(original_batch, num_candidates)[torch.arange(original_batch, device=steps.device), best]

        if self.clip_pred:
            trajectory = trajectory.clip(self.x_min, self.x_max)
        return trajectory, {"avg_steps": float(steps.float().mean().item())}
