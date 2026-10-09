import torch
from abc import ABC, abstractmethod


class BaseStopper(ABC):
    """Abstract base class for adaptive early-stopping policies."""

    def __init__(self, **kwargs):
        self.state = {}

    def reset_state(self, initial_trajectory: torch.Tensor):
        """Reset any tracked state at the start of sampling.

        For example, the previous trajectory used by the stall criterion.
        """
        self.state = {}

    @abstractmethod
    def check(self,
              update: torch.Tensor,
              trajectory: torch.Tensor,
              active_mask: torch.Tensor,
              ) -> torch.Tensor:
        """Decide which samples in the batch have converged.

        Args:
            update: per-sample update vector returned by the sampler.
            trajectory: current trajectory (x_t) BEFORE the update is applied.
            active_mask: (B,) boolean mask of samples still being refined.

        Returns:
            (B,) boolean mask. Samples that should stop are set to False.
        """
        pass
