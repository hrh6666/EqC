from .base_stopper import BaseStopper
from .null_stopper import NullStopper


def create_stopper(stopper_config: dict) -> BaseStopper:
    """Build the fixed-budget stopper used by EqC."""
    name = (stopper_config or {}).get("name", "null").lower()
    if name == "null":
        return NullStopper()
    raise ValueError(f"Unsupported stopper: {name}; this release supports only 'null'")


__all__ = ["BaseStopper", "NullStopper", "create_stopper"]
