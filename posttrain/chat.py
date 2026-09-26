"""Discute avec LlmEdith dans le terminal (depuis un export HF, avec cache KV via transformers).

    python -m posttrain.chat exports/llmedith-chat
    python -m posttrain.chat exports/llmedith-main --base      # modèle de base : simple complétion de texte
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--base", action="store_true", help="modèle non aligné : complétion au lieu de chat")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--system", default="You are Edith, a helpful and concise assistant.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.path)
    model = AutoModelForCausalLM.from_pretrained(args.path, dtype=torch.bfloat16 if device != "cpu" else torch.float32)
    model.to(device).eval()
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    history = [{"role": "system", "content": args.system}] if args.system and not args.base else []
    print("LlmEdith prêt. Ctrl+C pour quitter, /reset pour effacer la conversation.\n")
    while True:
        try:
            user = input("Toi > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user == "/reset":
            history = history[:1]
            continue
        if args.base:
            ids = tok(user, return_tensors="pt").input_ids.to(device)
        else:
            history.append({"role": "user", "content": user})
            ids = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True)["input_ids"].to(device)
        print("Edith > ", end="", flush=True)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                                 temperature=args.temperature, top_p=0.9, repetition_penalty=1.1,
                                 streamer=streamer, pad_token_id=tok.pad_token_id)
        reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        if not args.base:
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
