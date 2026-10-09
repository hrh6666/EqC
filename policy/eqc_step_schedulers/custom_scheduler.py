from typing import List

import torch

from .base_scheduler import BaseStepScheduler


class CustomStepScheduler(BaseStepScheduler):
    """Returns the explicit per-step schedule supplied by the configuration."""

    def __init__(self, num_steps: int, stepsize_list: List[float]):
        super().__init__(num_steps=num_steps, base_stepsize=stepsize_list[0])
        self.stepsize_list = stepsize_list

    def get_schedule(self) -> List[float]:
        
        schedule = torch.full(
            (self.num_steps,), self.base_stepsize, dtype=torch.float32
        )
        length = min(len(self.stepsize_list), self.num_steps)
        schedule[:length] = torch.tensor(self.stepsize_list[:length], dtype=torch.float32)  
        schedule[length:] = self.stepsize_list[-1]
        return schedule.tolist()
