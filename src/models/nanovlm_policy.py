from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.nano_imports import add_nanovlm_to_path


class NanoVLMPolicy:
    """Wraps nanoVLM for SFT / GRPO training on MiniGrid.

    Key design choice: we force **single-patch mode** by setting
    ``max_img_size = vit_img_size``.  MiniGrid frames are tiny (~96 px),
    so multi-patch splitting is pointless and causes tokenizer / embedding
    crashes with nanoVLM-222M (vit_img_size 224, grid tokens only up to 8×8).
    """

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

        # ---------- Single-patch mode ----------
        # Force max_img_size == vit_img_size so every image becomes exactly
        # one 224×224 (or 512×512) patch.  Grid = (1,1), no r/c tokens,
        # no embedding-size mismatches.
        patch = self.model.cfg.vit_img_size
        self.model.cfg.max_img_size = patch
        self.model.cfg.resize_to_max_side_len = False

        self.tokenizer = get_tokenizer(
            self.model.cfg.lm_tokenizer,
            self.model.cfg.vlm_extra_tokens,
            self.model.cfg.lm_chat_template,
        )

        # Make sure token embeddings cover all VLM-special tokens
        self._ensure_vocab_alignment()

        self.image_processor = get_image_processor(
            self.model.cfg.max_img_size,
            self.model.cfg.vit_img_size,
            self.model.cfg.resize_to_max_side_len,
        )

    # ------------------------------------------------------------------
    def _ensure_vocab_alignment(self) -> None:
        """Resize embedding / head if tokenizer has more tokens than model."""
        embed = self.model.decoder.token_embedding
        model_vocab = embed.num_embeddings
        tok_vocab = len(self.tokenizer)
        if tok_vocab <= model_vocab:
            return

        hidden = embed.embedding_dim
        device = embed.weight.device
        dtype = embed.weight.dtype

        new_embed = torch.nn.Embedding(tok_vocab, hidden, device=device, dtype=dtype)
        with torch.no_grad():
            new_embed.weight[:model_vocab].copy_(embed.weight)
            filler = embed.weight.mean(dim=0, keepdim=True)
            new_embed.weight[model_vocab:].copy_(filler.expand(tok_vocab - model_vocab, -1))
        self.model.decoder.token_embedding = new_embed

        if hasattr(self.model.decoder, "head"):
            old_head = self.model.decoder.head
            new_head = torch.nn.Linear(hidden, tok_vocab, bias=False, device=device, dtype=dtype)
            with torch.no_grad():
                rows = min(old_head.weight.shape[0], model_vocab)
                new_head.weight[:rows].copy_(old_head.weight[:rows])
                if tok_vocab > rows:
                    new_head.weight[rows:].copy_(new_embed.weight[rows:])
            self.model.decoder.head = new_head

        if getattr(self.model.cfg, "lm_tie_weights", False) and hasattr(self.model.decoder, "head"):
            self.model.decoder.head.weight = self.model.decoder.token_embedding.weight

        self.model.cfg.lm_vocab_size = tok_vocab

    # ------------------------------------------------------------------
    def set_mode(self, train: bool) -> None:
        self.model.train(train)

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, nanovlm_repo: str, device: str = "cuda") -> "NanoVLMPolicy":
        return cls(model_source=checkpoint_path, nanovlm_repo=nanovlm_repo, device=device)

    # ------------------------------------------------------------------
    def _encode_sample(self, frame: np.ndarray, user_prompt: str, assistant_text: str):
        image = Image.fromarray(frame)
        processed_images, split_ratio = self.image_processor(image)

        # In single-patch mode split_ratio is (1,1), processed_images has
        # shape [2, C, H, W] (global + 1 split) or [1, C, H, W].
        # If there is no global_image_token we keep only the split patches.
        if (
            not hasattr(self.tokenizer, "global_image_token")
            and split_ratio[0] * split_ratio[1] == len(processed_images) - 1
        ):
            processed_images = processed_images[1:]

        image_string = self._get_image_string(
            self.tokenizer, [split_ratio], self.model.cfg.mp_image_token_length
        )

        user_messages = [{"role": "user", "content": image_string + user_prompt}]
        full_messages = [
            {"role": "user", "content": image_string + user_prompt},
            {"role": "assistant", "content": assistant_text},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")
        full_ids = self.tokenizer(full_text, return_tensors="pt")

        input_ids = full_ids["input_ids"]
        attention_mask = full_ids["attention_mask"]
        labels = input_ids.clone()
        labels[:, : prompt_ids["input_ids"].shape[1]] = -100

        # Clamp token ids to valid embedding range (safety net)
        vocab_size = self.model.decoder.token_embedding.num_embeddings
        input_ids = input_ids.clamp(0, vocab_size - 1)

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
            "images": processed_images.to(self.device),
        }

    # ------------------------------------------------------------------
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

        image_string = self._get_image_string(
            self.tokenizer, [split_ratio], self.model.cfg.mp_image_token_length
        )
        messages = [{"role": "user", "content": image_string + user_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt_text, return_tensors="pt")

        # Clamp token ids (safety net)
        vocab_size = self.model.decoder.token_embedding.num_embeddings
        input_ids = encoded["input_ids"].clamp(0, vocab_size - 1)

        generated_ids = self.model.generate(
            input_ids=input_ids.to(self.device),
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
