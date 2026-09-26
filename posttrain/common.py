"""Outils communs au SFT et au DPO : format de chat ChatML, masques de loss, chargement et sauvegarde."""
from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

from llmedith.config import ModelConfig
from llmedith.torch_impl.model import LlmEdith
from llmedith.torch_impl.muon import native_bf16
from llmedith.torch_impl.train import load_model_state, model_state


class ChatFormatter:
    """<|im_start|>role\\ncontenu<|im_end|>\\n : la loss ne porte que sur les réponses de l'assistant."""

    def __init__(self, tokenizer_path: str):
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.im_start = self.tok.token_to_id("<|im_start|>")
        self.im_end = self.tok.token_to_id("<|im_end|>")
        self.pad = self.tok.token_to_id("<|pad|>")
        self.nl = self.tok.encode("\n").ids

    def encode(self, messages: list[dict], add_generation_prompt: bool = False) -> tuple[list[int], list[int]]:
        ids, mask = [], []
        for m in messages:
            head = [self.im_start] + self.tok.encode(f"{m['role']}\n").ids
            body = self.tok.encode(m["content"]).ids + [self.im_end]
            train = 1 if m["role"] == "assistant" else 0
            ids += head + body + self.nl
            mask += [0] * len(head) + [train] * len(body) + [0] * len(self.nl)
        if add_generation_prompt:
            head = [self.im_start] + self.tok.encode("assistant\n").ids
            ids += head
            mask += [0] * len(head)
        return ids, mask


def load_checkpoint(ckpt_dir: str, device) -> tuple[LlmEdith, dict]:
    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    model = LlmEdith(ModelConfig(**meta["config"]["model"]))
    load_model_state(model, os.path.join(ckpt_dir, "model.safetensors"))
    return model.to(device), meta


def save_checkpoint(ckpt_dir: str, model: LlmEdith, meta: dict, step: int, stage: str) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    save_file(model_state(model), os.path.join(ckpt_dir, "model.safetensors"))
    meta = dict(meta, step=step, stage=stage, framework="torch-posttrain", loader=None)
    json.dump(meta, open(os.path.join(ckpt_dir, "meta.json"), "w"), indent=2)


def pick_device_dtype():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        return dev, (torch.bfloat16 if native_bf16(dev) else torch.float16)
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    return torch.device("cpu"), torch.float32
