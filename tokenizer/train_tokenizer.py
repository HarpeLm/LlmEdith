"""Entraîne le tokenizer BPE byte-level de LlmEdith (32 768 tokens) sur FineWeb-Edu (+10 % de code).

- Découpage des mots façon GPT-4/Llama 3, mais chiffres isolés un par un (meilleur en arithmétique
  pour les petits modèles, comme SmolLM).
- Byte-level : aucun caractère inconnu possible.
- 32 768 tokens : tient en uint16, et petit vocabulaire donc moins de paramètres d'embedding.

    python -m tokenizer.train_tokenizer --num_bytes 5e9
"""
from __future__ import annotations

import argparse
import os

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]
SPLIT_PATTERN = (r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*"
                 r"|\s*[\r\n]|\s+(?!\S)|\s+")


def build_tokenizer() -> tuple[Tokenizer, trainers.BpeTrainer]:
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def text_iterator(sources: list[tuple[str, str | None, str, float]], num_bytes: float):
    """Mélange plusieurs datasets (streaming) selon leurs fractions en octets."""
    from datasets import load_dataset
    its = [iter(load_dataset(p, c, split="train", streaming=True)) for p, c, _, _ in sources]
    seen = [0] * len(sources)
    while sum(seen) < num_bytes:
        i = min(range(len(sources)), key=lambda k: seen[k] / sources[k][3])
        text = next(its[i])[sources[i][2]]
        seen[i] += len(text.encode("utf-8"))
        yield text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--field", default="text")
    ap.add_argument("--code_dataset", default="codeparrot/codeparrot-clean")
    ap.add_argument("--code_field", default="content")
    ap.add_argument("--code_frac", type=float, default=0.1, help="part de code (indentation, symboles)")
    ap.add_argument("--num_bytes", type=float, default=5e9)
    ap.add_argument("--vocab_size", type=int, default=32768)
    ap.add_argument("--out", default="tokenizer/llmedith-32k")
    args = ap.parse_args()

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size, special_tokens=SPECIAL_TOKENS, min_frequency=2, show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    sources = [(args.dataset, args.config, args.field, 1 - args.code_frac)]
    if args.code_frac > 0:
        sources.append((args.code_dataset, None, args.code_field, args.code_frac))
    tok.train_from_iterator(text_iterator(sources, args.num_bytes), trainer=trainer)
    os.makedirs(args.out, exist_ok=True)
    tok.save(os.path.join(args.out, "tokenizer.json"))
    for s in SPECIAL_TOKENS:
        print(s, tok.token_to_id(s))
    sample = "LlmEdith learns English: 12345 apples, 3.14159 pies!\n    def f(x): return x**2"
    enc = tok.encode(sample)
    print(f"{len(enc.ids)} tokens : {enc.tokens}")
    assert tok.decode(enc.ids) == sample
    print(f"Tokenizer sauvegardé dans {args.out}/tokenizer.json (vocab {tok.get_vocab_size()})")


if __name__ == "__main__":
    main()
