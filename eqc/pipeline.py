"""Training and standard LIBERO evaluation helpers for no-consistency EqC."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import gym
import numpy as np
import pytorch_lightning as L
import torch
import torch.multiprocessing as torch_mp
from cleandiffuser.dataset.libero_dataset import LiberoDataset
from cleandiffuser.nn_diffusion import DiT1dWithACICrossAttention
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import T5Config

from policy.observation import build_observation_transforms, process_observation
from policy.vision_language import EqCConditionEncoder
from utils import DatasetWrapper, T5LanguageEncoder

from .model import EqCModel


def resolve_paths(cfg: DictConfig) -> Tuple[Path, Path, str]:
    dataset_root = Path(to_absolute_path(cfg.paths.dataset_root))
    return (
        dataset_root / f"{cfg.task_suite}.zarr",
        Path(to_absolute_path(cfg.paths.save_root)) / cfg.task_suite,
        cfg.task_suite.replace("_", "-") + "-v0",
    )


def create_dataset(dataset_path: Path, To: int, Ta: int) -> LiberoDataset:
    return LiberoDataset(data_path=dataset_path, observation_meta=["color", "color_ego"], To=To, Ta=Ta)


def create_dataloaders(cfg: DictConfig, dataset: LiberoDataset, devices: Tuple[int, ...]) -> torch.utils.data.DataLoader:
    workers = int(cfg.training.dataloader.num_workers)
    kwargs: Dict[str, Any] = {
        "batch_size": int(cfg.training.dataloader.batch_size) // max(len(devices), 1),
        "shuffle": bool(cfg.training.dataloader.shuffle),
        "num_workers": workers,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(cfg.training.dataloader.persistent_workers)
    return torch.utils.data.DataLoader(DatasetWrapper(dataset), **kwargs)


def create_policy(cfg: DictConfig, To: int, Ta: int, act_dim: int) -> EqCModel:
    t5_hidden_dim = T5Config.from_pretrained(cfg.pretrained.t5).d_model
    return EqCModel(
        nn_diffusion=DiT1dWithACICrossAttention(
            x_dim=act_dim,
            x_seq_len=Ta,
            emb_dim=768,
            d_model=384,
            n_heads=6,
            depth=12,
            timestep_emb_type="untrainable_fourier",
            timestep_emb_params={"scale": 0.2},
        ),
        nn_condition=EqCConditionEncoder(
            vit_path=cfg.pretrained.vit, t5_hidden_dim=t5_hidden_dim, emb_dim=768, freeze=True, To=To, n_views=2
        ),
        x_max=torch.full((Ta, act_dim), 1.0),
        x_min=torch.full((Ta, act_dim), -1.0),
        per_action_noise=bool(cfg.policy.per_action_noise),
    )


def run_training(
    cfg: DictConfig,
    policy: EqCModel,
    train_loader: torch.utils.data.DataLoader,
    output_dir: Path,
    devices: Tuple[int, ...],
    ckpt_path: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = None
    if cfg.wandb.enabled:
        os.environ["WANDB_MODE"] = str(cfg.wandb.mode)
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            name=cfg.wandb.name,
            tags=list(cfg.wandb.tags),
            log_model=cfg.wandb.log_model,
            save_dir=str(output_dir),
        )
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    checkpoint_dir = output_dir / "checkpoints" if logger is None else Path(logger.experiment.dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        every_n_train_steps=int(cfg.training.checkpoint_interval),
        save_top_k=-1,
        filename="{step}",
    )
    trainer = L.Trainer(
        devices=devices,
        max_steps=int(cfg.training.steps),
        callbacks=[checkpoint_callback],
        default_root_dir=str(output_dir),
        precision=cfg.training.precision,
        strategy=cfg.training.strategy,
        accumulate_grad_batches=int(cfg.training.accumulate_grad_batches),
        logger=logger if logger is not None else False,
    )
    trainer.fit(policy, train_loader, ckpt_path=ckpt_path)


def _task_ids(cfg: DictConfig) -> List[int]:
    requested = cfg.inference.task_ids
    if isinstance(requested, str) and requested.lower() == "all":
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[cfg.task_suite]()
        # LIBERO releases expose the suite size under different names.
        num_tasks = getattr(suite, "num_tasks", None)
        if num_tasks is None:
            num_tasks = getattr(suite, "n_tasks", None)
        if num_tasks is None and hasattr(suite, "get_num_tasks"):
            num_tasks = suite.get_num_tasks()
        if num_tasks is None and hasattr(suite, "get_task_names"):
            num_tasks = len(suite.get_task_names())
        if num_tasks is None:
            raise AttributeError(f"Unable to determine the number of tasks for suite '{cfg.task_suite}'")
        return list(range(int(num_tasks)))
    if isinstance(requested, ListConfig):
        requested = list(requested)
    if isinstance(requested, (list, tuple)):
        return [int(task_id) for task_id in requested]
    return [int(requested)]


def _normalizer_spec(dataset: LiberoDataset, act_dim: int) -> Dict[str, Any]:
    normalizer = dataset.get_normalizer()["action"]
    try:
        mean, std = getattr(normalizer, "mean"), getattr(normalizer, "std")
        return {"kind": "zscore", "mean": np.asarray(mean, dtype=np.float32), "std": np.asarray(std, dtype=np.float32)}
    except AttributeError:
        zeros, ones = np.zeros((1, 1, act_dim), dtype=np.float32), np.ones((1, 1, act_dim), dtype=np.float32)
        bias = normalizer.unnormalize(zeros)[0, 0]
        scale = normalizer.unnormalize(ones)[0, 0] - bias
        return {"kind": "affine", "bias": bias, "scale": scale}


def _unnormalize(actions: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    if spec["kind"] == "zscore":
        return actions * spec["std"][None, None, :] + spec["mean"][None, None, :]
    return actions * spec["scale"][None, None, :] + spec["bias"][None, None, :]


@dataclass(frozen=True)
class _WorkerInput:
    cfg_dict: dict
    task_id: int
    device_id: int
    env_name: str
    To: int
    Ta: int
    act_dim: int
    num_act_exec: int
    max_episode_steps: int
    normalizer: Dict[str, Any]


def _evaluate_task(worker: _WorkerInput, policy_factory: Callable[[], EqCModel]) -> Dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch_mp.set_sharing_strategy("file_system")
    cfg = OmegaConf.create(worker.cfg_dict)
    device = torch.device(f"cuda:{worker.device_id}")
    image_norm, center_crop = build_observation_transforms(tuple(cfg.constants.norm_params))
    language_encoder = T5LanguageEncoder(cfg.pretrained.t5, max_length=32, device=device)
    policy = policy_factory().to(device).eval()
    checkpoint = torch.load(to_absolute_path(cfg.inference.ckpt_file), map_location="cpu", weights_only=True)
    policy.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    prior = torch.zeros((1, worker.Ta, worker.act_dim), device=device)
    env = gym.make(
        worker.env_name,
        task_id=worker.task_id,
        image_size=224,
        require_depth=False,
        require_point_cloud=False,
        seed=cfg.seed,
        max_episode_steps=worker.max_episode_steps,
    )
    successes: List[bool] = []
    replan_counts: List[int] = []
    try:
        description = env.task_description
        language_embedding, language_mask = language_encoder([description])
        trials = env.num_init_states
        if cfg.inference.max_init_states is not None:
            trials = min(trials, int(cfg.inference.max_init_states))
        print(f"Task {worker.task_id}: {description}")
        for trial in range(trials):
            observation, done, reward_sum = env.reset(init_state_id=trial), False, 0
            image, image_ego = process_observation(
                observation,
                device=device,
                To=worker.To,
                img_norm=image_norm,
                center_crop=center_crop,
            )
            image = torch.cat([image, image_ego], dim=1)
            dummy_steps, replans = 0, 0
            while not np.all(done):
                action, _ = policy.sample(
                    prior,
                    condition_cfg={"image": image, "language_embedding": language_embedding, "language_mask": language_mask},
                    use_ema=False,
                    temperature=float(cfg.inference.temperature),
                    sampler_config=cfg.sampler_config,
                    stopper_config=cfg.stopper_config,
                    scheduler_config=cfg.scheduler_config,
                    num_candidates=int(cfg.inference.num_candidates),
                )
                action_np = _unnormalize(action.detach().cpu().numpy(), worker.normalizer)
                if dummy_steps < int(cfg.inference.num_dummy_steps):
                    action_np = np.zeros((1, worker.num_act_exec, worker.act_dim), dtype=np.float32)
                    dummy_steps += 1
                replans += 1
                for index in range(worker.num_act_exec):
                    next_observation, reward, step_done, _ = env.step(action_np[0, index])
                    done, reward_sum = np.logical_or(done, step_done), reward_sum + reward
                    if index >= worker.num_act_exec - worker.To:
                        new_image, new_image_ego = process_observation(
                            next_observation,
                            device=device,
                            To=worker.To,
                            img_norm=image_norm,
                            center_crop=center_crop,
                        )
                        image[:, index - worker.num_act_exec + worker.To] = new_image[:, 0]
                        image[:, index - worker.num_act_exec + 2 * worker.To] = new_image_ego[:, 0]
                    if np.all(done):
                        break
                observation = next_observation
            success = bool(reward_sum)
            successes.append(success)
            replan_counts.append(replans)
            print(f"Task {worker.task_id} trial {trial}: success={success}")
    finally:
        env.close()
    return {
        "task_id": worker.task_id,
        "successes": int(sum(successes)),
        "trials": len(successes),
        "mean_replans": float(np.mean(replan_counts)) if replan_counts else 0.0,
    }


def run_inference_parallel(
    cfg: DictConfig,
    policy_factory: Callable[[], EqCModel],
    dataset: LiberoDataset,
    env_name: str,
    *,
    To: int,
    Ta: int,
    act_dim: int,
    num_act_exec: int,
    max_episode_steps: int,
) -> None:
    try:
        torch_mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    devices = [int(device) for device in cfg.devices]
    workers = [
        _WorkerInput(
            cfg_dict=cfg_dict,
            task_id=task_id,
            device_id=devices[index % len(devices)],
            env_name=env_name,
            To=To,
            Ta=Ta,
            act_dim=act_dim,
            num_act_exec=num_act_exec,
            max_episode_steps=max_episode_steps,
            normalizer=_normalizer_spec(dataset, act_dim),
        )
        for index, task_id in enumerate(_task_ids(cfg))
    ]
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(cfg.inference.max_workers), mp_context=torch_mp.get_context("spawn")) as executor:
        futures = [executor.submit(_evaluate_task, worker, policy_factory) for worker in workers]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Task {result['task_id']}: {result['successes']}/{result['trials']} successful")
    total_successes, total_trials = sum(row["successes"] for row in results), sum(row["trials"] for row in results)
    print(f"Overall success rate: {total_successes}/{total_trials} = {total_successes / total_trials:.4f}")
