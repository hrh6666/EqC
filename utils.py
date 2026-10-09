from typing import List

import numpy as np
import torch
import torchvision.transforms as T


class DatasetWrapper:
    def __init__(self, dataset, no_ego=False):
        NORM_PARAMS = (0.5, 0.5, 0.5)
        self.dataset = dataset
        self.no_ego = no_ego
        # add normalization and random crop to 3rd person view
        # add only normalization to egocentric view
        self.normalize = T.Normalize(NORM_PARAMS, NORM_PARAMS)
        self.vis_aug = T.Compose([T.Normalize(NORM_PARAMS, NORM_PARAMS), T.RandomCrop(200), T.Resize(224)])

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Rescale to [0, 1] and apply random crop and resize
        image = item["observation"]["color"].astype(np.float32) / 255.0
        image = self.vis_aug(torch.tensor(image))

        # Use hand-view image
        if "color_ego" in item["observation"].keys():
            image_ego = item["observation"]["color_ego"].astype(np.float32) / 255.0
            image_ego = self.normalize(torch.tensor(image_ego))
            combined_image = torch.cat([image, image_ego], dim=0)
        else:
            combined_image = image

        act = item["action"]
        obs = {
            "image": combined_image,  # (n_view, 3, 224, 224)
            "language_embedding": item["language_embedding"],
            "language_mask": item["language_mask"],
        }
        # Use low-dim state
        if "eef_states" in item["observation"].keys():
            eef_states = item["observation"]["eef_states"].astype(np.float32)
            obs["eef_states"] = eef_states

        return {"x0": act, "condition_cfg": obs}


class T5LanguageEncoder:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "google-t5/t5-base",
        max_length: int = 32,
        device: str = "cpu",
    ):
        from transformers import T5EncoderModel, T5Tokenizer

        self.device = device
        self.max_length = max_length
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = T5EncoderModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

    @torch.no_grad()
    def encode(self, sentences: List[str]):
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)
        return last_hidden_states, attention_mask

    def __call__(self, sentences: List[str]):
        return self.encode(sentences)
