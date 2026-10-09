import torch

from .base_stopper import BaseStopper


class NullStopper(BaseStopper):
    """Stopper that never fires; the loop runs for the full step budget.

    Equivalent to adaptive=False.
    """

    def check(self,
              update: torch.Tensor,
              trajectory: torch.Tensor,
              active_mask: torch.Tensor,
              ) -> torch.Tensor:
        # Return the input mask unchanged.
        return active_mask
