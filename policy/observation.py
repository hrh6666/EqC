"""Utility helpers for preparing visual observations."""
from __future__ import annotations

from typing import Tuple

import torch
import torchvision.transforms as T


def build_observation_transforms(norm_params: Tuple[float, float, float]) -> Tuple[T.Normalize, T.Compose]:
    """Construct the transforms used for agent-view and wrist camera images."""

    norm_params = tuple(norm_params)
    img_norm = T.Normalize(norm_params, norm_params)
    center_crop = T.Compose([T.Normalize(norm_params, norm_params), T.CenterCrop(200), T.Resize(224)])
    return img_norm, center_crop


import numpy as np
import matplotlib.pyplot as plt


def _to_hwc_for_plot(img: np.ndarray) -> np.ndarray:
    """Convert image to HWC layout for matplotlib."""
    if img.ndim == 3:
        # CHW -> HWC
        if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))
    return img

def mask_center_30_percent(image: torch.Tensor) -> torch.Tensor:
    """
    image: torch.Tensor of shape (C, H, W), value range [0, 1]
    return: same shape, center 30% region set to 0
    """
    assert image.ndim == 3, "Expect image shape (C, H, W)"

    _, H, W = image.shape
    h = int(0.5 * H)
    w = int(0.5 * W)

    y1 = (H - h) // 2
    y2 = y1 + h
    x1 = (W - w) // 2
    x2 = x1 + w

    image = image.clone()  # avoid in-place side effects
    image[:, y1:y2, x1:x2] = 0.0
    return image

def process_observation(
    obs: dict,
    *,
    device: torch.device,
    To: int,
    img_norm: T.Normalize,
    center_crop: T.Compose,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = obs["agentview_image"] / 255.0
    image = torch.tensor(image, device=device, dtype=torch.float32)
    image = center_crop(image)
    # image = mask_center_30_percent(image)
    image = image[None, None].repeat(1, To, 1, 1, 1)

    image_ego = obs["robot0_eye_in_hand_image"] / 255.0
    image_ego = torch.tensor(image_ego, device=device, dtype=torch.float32)
    # image_ego = mask_center_30_percent(image_ego)
    image_ego = img_norm(image_ego)
    image_ego = image_ego[None, None].repeat(1, To, 1, 1, 1)

    return image, image_ego