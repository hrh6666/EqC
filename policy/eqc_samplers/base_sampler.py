import torch
from abc import ABC, abstractmethod
from typing import Optional


class BaseSampler(ABC):
    """Abstract base class for refinement-step samplers / optimizers."""

    def __init__(self):
        self.state = {}  # holds optimizer state such as momentum (m), second moment (v)

    def reset_state(self, initial_trajectory: torch.Tensor):
        """Initialize or reset optimizer state at the start of sampling.

        Args:
            initial_trajectory: initial action chunk; used to read shape, device,
                and dtype for state buffers.
        """
        self.state = {}

    @abstractmethod
    def step(
            self, stepsize,
            trajectory: torch.Tensor,
            model: callable,
            global_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform one refinement step.

        Args:
            stepsize: per-step learning rate eta.
            trajectory: current action chunk x_t.
            model: velocity-field network f_theta predicting v_theta(x_t, cond).
            global_cond: encoded observation conditioning.

        Returns:
            The update vector for this step (typically stepsize * velocity).
            The outer loop applies trajectory = trajectory + update.
        """
        pass
