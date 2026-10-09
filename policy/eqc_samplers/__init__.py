from .base_sampler import BaseSampler
from .gd_sampler import GDSampler


def create_sampler(
    config: dict = None,
) -> BaseSampler:
    """Build the gradient-descent sampler used by EqC."""
    sampler_name = (config or {}).get("name", "gd").lower()
    if sampler_name == "gd":
        return GDSampler()
    raise ValueError(f"Unsupported sampler: {sampler_name}; this release supports only 'gd'")


__all__ = ["BaseSampler", "GDSampler", "create_sampler"]
