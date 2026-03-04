from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.nano_imports import add_nanovlm_to_path


class NanoVLMPolicy:
    def __init__(
        self,
        model_source: str,
        nanovlm_repo: str,
        device: str = "cuda",
    ) -> None:
        add_nanovlm_to_path(nanovlm_repo)

        from models.vision_language_model import VisionLanguageModel
        from data.processors import get_image_processor, get_image_string, get_tokenizer

        self._get_image_string = get_image_string
        self.model = VisionLanguageModel.from_pretrained(model_source)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.train()

        self.tokenizer = get_tokenizer(
            self.model.cfg.lm_tokenizer,
            self.model.cfg.vlm_extra_tokens,
            self.model.cfg.lm_chat_template,
        )

        resize_to_max_side_len = getattr(self.model.cfg, "resize_to_max_side_len", False)
        self.image_processor = get_image_processor(
            self.model.cfg.max_img_size,
            self.model.cfg.vit_img_size,
            resize_to_max_side_len,
        )

    def set_mode(self, train: bool) -> None:
        self.model.train(train)

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, nanovlm_repo: str, device: str = "cuda") -> "NanoVLMPolicy":
        return cls(model_source=checkpoint_path, nanovlm_repo=nanovlm_repo, device=device)

    def _encode_sample(self, frame: np.ndarray, user_prompt: str, assistant_text: str):
        image = Image.fromarray(frame)
        processed_images, split_ratio = self.image_processor(image)
        if (
            not hasattr(self.tokenizer, "global_image_token")
            and split_ratio[0] * split_ratio[1] == len(processed_images) - 1
        ):
            processed_images = processed_images[1:]

        image_string = self._get_image_string(self.tokenizer, [split_ratio], self.model.cfg.mp_image_token_length)

        user_messages = [{"role": "user", "content": image_string + user_prompt}]
        full_messages = [
            {"role": "user", "content": image_string + user_prompt},
            {"role": "assistant", "content": assistant_text},
        ]

        prompt_text = self.tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
        full_text = self.tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")
        full_ids = self.tokenizer(full_text, return_tensors="pt")

        input_ids = full_ids["input_ids"]
        attention_mask = full_ids["attention_mask"]
        labels = input_ids.clone()
        labels[:, : prompt_ids["input_ids"].shape[1]] = -100

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
            "images": processed_images.to(self.device),
        }

    def sft_loss(self, frame: np.ndarray, user_prompt: str, assistant_text: str) -> torch.Tensor:
        batch = self._encode_sample(frame, user_prompt, assistant_text)
        _logits, loss = self.model(
            batch["input_ids"],
            batch["images"],
            attention_mask=batch["attention_mask"],
            targets=batch["labels"],
        )
        return loss

    @torch.no_grad()
    def generate(self, frame: np.ndarray, user_prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        self.model.eval()
        image = Image.fromarray(frame)
        processed_images, split_ratio = self.image_processor(image)
        if (
            not hasattr(self.tokenizer, "global_image_token")
            and split_ratio[0] * split_ratio[1] == len(processed_images) - 1
        ):
            processed_images = processed_images[1:]

        image_string = self._get_image_string(self.tokenizer, [split_ratio], self.model.cfg.mp_image_token_length)
        messages = [{"role": "user", "content": image_string + user_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt_text, return_tensors="pt")

        generated_ids = self.model.generate(
            input_ids=encoded["input_ids"].to(self.device),
            images=processed_images.to(self.device),
            attention_mask=encoded["attention_mask"].to(self.device),
            max_new_tokens=max_new_tokens,
            greedy=temperature <= 1e-8,
            temperature=max(temperature, 1e-6),
        )
        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.model.train()
        return text.strip()


def parse_action(text: str) -> Optional[str]:
    action_line = re.search(r"ACTION\s*:\s*(left|right|forward)", text, re.IGNORECASE)
    if action_line:
        return action_line.group(1).lower()
    any_action = re.search(r"\b(left|right|forward)\b", text, re.IGNORECASE)
    if any_action:
        return any_action.group(1).lower()
    return None
