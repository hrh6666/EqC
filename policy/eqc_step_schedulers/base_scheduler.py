from abc import ABC, abstractmethod
from typing import List


class BaseStepScheduler(ABC):
    """Abstract base class for step-size schedulers."""

    def __init__(self, num_steps: int, base_stepsize: float):
        self.num_steps = int(num_steps)
        self.base_stepsize = float(base_stepsize)

    @abstractmethod
    def get_schedule(self) -> List[float]:
        """Return a list of step sizes to use for each iteration."""
        raise NotImplementedError
