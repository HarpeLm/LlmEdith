"""Exporte un checkpoint LlmEdith (PyTorch ou JAX) au format Hugging Face `Qwen3ForCausalLM`.

Une fois exporté, le modèle marche directement avec transformers, lm-eval-harness, vLLM et llama.cpp.

    python -m convert.to_hf --ckpt runs/main/latest --out exports/llmedith-main
    python -m convert.to_hf --ckpt runs/sft/latest --out exports/llmedith-chat --push MonPseudo/llmedith-chat
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from llmedith.config import ModelConfig

CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]


def hf_config(mc: ModelConfig, eos_id: int = 0, pad_id: int = 3) -> dict:
    return dict(
        vocab_size=mc.vocab_size, hidden_size=mc.d_model, intermediate_size=mc.ffn_dim,
        num_hidden_layers=mc.n_layers, num_attention_heads=mc.n_heads, num_key_value_heads=mc.n_kv_heads,
        head_dim=mc.head_dim, max_position_embeddings=mc.max_seq_len, rope_theta=mc.rope_theta,
        rope_parameters={"rope_type": "default", "rope_theta": mc.rope_theta},
        rms_norm_eps=mc.norm_eps, tie_word_embeddings=mc.tie_embeddings, attention_bias=False,
        hidden_act="silu", use_sliding_window=False, bos_token_id=eos_id, eos_token_id=eos_id,
        pad_token_id=pad_id, initializer_range=mc.init_std,
    )


def save_tokenizer(tokenizer_json: str, out_dir: str, eos: str = "<|endoftext|>") -> None:
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json, eos_token=eos, bos_token=eos,
                                  pad_token="<|pad|>", unk_token=None,
                                  additional_special_tokens=["<|im_start|>", "<|im_end|>"])
    tok.chat_template = CHAT_TEMPLATE
    tok.model_max_length = 1_000_000
    tok.save_pretrained(out_dir)


def export(ckpt_dir: str, out_dir: str, tokenizer_json: str, chat: bool = False) -> str:
    import torch
    from safetensors.torch import load_file, save_file
    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    mc = ModelConfig(**meta["config"]["model"])
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in load_file(os.path.join(ckpt_dir, "model.safetensors")).items()}
    if mc.tie_embeddings:
        sd.pop("lm_head.weight", None)
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})

    from tokenizers import Tokenizer
    t = Tokenizer.from_file(tokenizer_json)
    eos = "<|im_end|>" if chat else "<|endoftext|>"
    cfg = hf_config(mc, eos_id=t.token_to_id(eos), pad_id=t.token_to_id("<|pad|>"))
    cfg.update(architectures=["Qwen3ForCausalLM"], model_type="qwen3", torch_dtype="bfloat16")
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    json.dump({"bos_token_id": cfg["bos_token_id"], "eos_token_id": cfg["eos_token_id"],
               "pad_token_id": cfg["pad_token_id"]}, open(os.path.join(out_dir, "generation_config.json"), "w"))
    save_tokenizer(tokenizer_json, out_dir, eos=eos)
    shutil.copy(os.path.join(ckpt_dir, "meta.json"), os.path.join(out_dir, "llmedith_meta.json"))
    print(f"Export HF terminé : {out_dir} (step {meta['step']})")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/llmedith-32k/tokenizer.json")
    ap.add_argument("--chat", action="store_true", help="modèle après SFT : EOS = <|im_end|>")
    ap.add_argument("--push", default="", help="dépôt HF où publier (privé)")
    args = ap.parse_args()
    out = export(args.ckpt, args.out, args.tokenizer, args.chat)
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, private=True, exist_ok=True)
        api.upload_folder(repo_id=args.push, folder_path=out)
        print(f"Publié (privé) : https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
