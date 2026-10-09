"""Shared ViT + T5 conditioning modules used by diffusion policies."""
from __future__ import annotations

from typing import Dict, Optional

import einops
import torch
import torch.nn as nn
from transformers import AutoModel

from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.utils import UntrainablePositionalEmbedding


class VisionLanguageEncoder(BaseNNCondition):
    """Conditioning module that fuses ViT and T5 features.

    Sub-classes can override :meth:`build_additional_outputs` to augment the
    returned dictionary with extra values.
    """

    def __init__(
        self,
        *,
        vit_path: str = "facebook/dinov2-base",
        t5_hidden_dim: int = 768,
        emb_dim: int = 384,
        freeze: bool = True,
        To: int = 1,
        n_views: int = 2,
    ) -> None:
        super().__init__()

        self.vit_model = AutoModel.from_pretrained(vit_path)
        if freeze:
            for param in self.vit_model.parameters():
                param.requires_grad = False

        vit_hidden_dim = self.vit_model.config.hidden_size
        self.num_hidden_layers = self.vit_model.config.num_hidden_layers

        self.vit_adapter = nn.Sequential(
            nn.Linear(vit_hidden_dim, emb_dim),
            nn.GELU(approximate="tanh"),
        )
        self.t5_adapter = nn.Sequential(
            nn.Linear(t5_hidden_dim, emb_dim),
            nn.GELU(approximate="tanh"),
        )
        self.To_pos_emb = nn.Parameter(torch.randn(1, To * n_views, 1, emb_dim) * 0.02)

        num_patches = (224 // self.vit_model.config.patch_size) ** 2
        vit_pos_emb = UntrainablePositionalEmbedding(emb_dim, 1000)(torch.arange(num_patches))
        vit_pos_emb = vit_pos_emb[None, None] * 0.2
        self.vit_pos_emb = nn.Parameter(vit_pos_emb)

        t5_pos_emb = UntrainablePositionalEmbedding(emb_dim, 100)(torch.arange(32))
        t5_pos_emb = t5_pos_emb[None] * 0.2
        self.t5_pos_emb = nn.Parameter(t5_pos_emb)

    def forward(
        self,
        condition: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        requires_attn_map: bool = False,
    ) -> Dict[str, torch.Tensor]:
        image = condition["image"]
        language_embedding = condition["language_embedding"]
        language_mask = condition["language_mask"]
        batch_size = image.shape[0]

        if language_mask is not None:
            language_mask = torch.logical_not(language_mask.to(torch.bool))

        image = einops.rearrange(image, "b To C H W -> (b To) C H W")
        vision_output = self.vit_model(image, output_attentions=requires_attn_map)
        vision_feat = vision_output["last_hidden_state"][:, 1:]
        vision_feat = self.vit_adapter(vision_feat)
        vision_feat = einops.rearrange(vision_feat, "(b To) m d -> b To m d", b=batch_size)
        vision_feat = vision_feat + self.To_pos_emb + self.vit_pos_emb
        vision_feat = einops.rearrange(vision_feat, "b To m d -> b (To m) d")

        lang_feat = self.t5_adapter(language_embedding) + self.t5_pos_emb

        output: Dict[str, torch.Tensor] = {
            "vec_condition": None,
            "lang_condition": lang_feat,
            "lang_condition_mask": language_mask,
            "vis_condition": vision_feat,
            "vis_condition_mask": None,
        }
        output.update(self.build_additional_outputs(vision_output, language_embedding, language_mask, batch_size))
        return output

    def build_additional_outputs(
        self,
        vision_output: Dict[str, torch.Tensor],
        language_embedding: torch.Tensor,
        language_mask: Optional[torch.Tensor],
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Hook for subclasses to append extra values to the output dict."""

        return {}


class EqCConditionEncoder(VisionLanguageEncoder):
    """Vanilla ViT + T5 condition used by the EqC policy."""

    pass
