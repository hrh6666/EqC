from .base_scheduler import BaseStepScheduler
from .custom_scheduler import CustomStepScheduler

def create_step_scheduler(
    config: dict,
    *,
    default_stepsize: float,
) -> BaseStepScheduler:
    """Build the explicit step-size scheduler used by EqC."""
    name = config["name"].lower()
    params = config["params"]
    num_steps = params["steps"]
    if name == "custom":
        return CustomStepScheduler(num_steps=num_steps, stepsize_list=params.get("stepsize_list", [default_stepsize]))
    raise ValueError(f"Unsupported scheduler: {name}; this release supports only 'custom'")


__all__ = [
    "BaseStepScheduler",
    "CustomStepScheduler",
    "create_step_scheduler",
]
