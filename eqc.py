"""Training and evaluation entry point for the no-consistency EqC policy."""
from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import hydra
from cleandiffuser.utils import set_seed
from omegaconf import DictConfig, ListConfig, OmegaConf

from eqc.pipeline import (
    create_dataloaders,
    create_dataset,
    create_policy,
    resolve_paths,
    run_inference_parallel,
    run_training,
)


def _as_tuple(devices) -> Tuple[int, ...]:
    if isinstance(devices, ListConfig):
        devices = list(devices)
    return tuple(devices) if isinstance(devices, (list, tuple)) else (int(devices),)


def make_policy_from_cfg(cfg_dict: dict, To: int, Ta: int, act_dim: int):
    return create_policy(OmegaConf.create(cfg_dict), To=To, Ta=Ta, act_dim=act_dim)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    dataset_path, output_dir, env_name = resolve_paths(cfg)
    To, Ta = cfg.constants.To, cfg.constants.Ta
    act_dim = 7
    max_episode_steps = 600 if "10" in cfg.task_suite or "goal" in cfg.task_suite else 300
    dataset = create_dataset(dataset_path, To, Ta)
    policy = create_policy(cfg, To=To, Ta=Ta, act_dim=act_dim)
    devices = _as_tuple(cfg.devices)

    if cfg.mode == "training":
        train_loader = create_dataloaders(cfg, dataset, devices)
        run_training(cfg, policy, train_loader, output_dir, devices, cfg.training.resume_ckpt)
    elif cfg.mode == "inference":
        policy_factory: Callable[[], object] = partial(
            make_policy_from_cfg, OmegaConf.to_container(cfg, resolve=True), To, Ta, act_dim
        )
        run_inference_parallel(
            cfg=cfg,
            policy_factory=policy_factory,
            dataset=dataset,
            env_name=env_name,
            To=To,
            Ta=Ta,
            act_dim=act_dim,
            num_act_exec=cfg.constants.num_act_exec,
            max_episode_steps=max_episode_steps,
        )
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
