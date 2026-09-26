"""Benchmarks de LlmEdith avec lm-evaluation-harness, et comparaison avec des modèles de référence.

    # nos modèles (exports HF) + les références publiques
    python -m eval.run_benchmarks --models edith-main=exports/llmedith-main --baselines
    # vérif rapide de la chaîne d'évaluation
    python -m eval.run_benchmarks --models gpt2=openai-community/gpt2 --limit 200

Résultats : results/<label>.json, plus un tableau récapitulatif dans results/benchmarks.md.
"""
from __future__ import annotations

import argparse
import json
import os

# tâche -> métrique retenue (0-shot, protocole type SmolLM / Pythia)
TASKS = {
    "hellaswag": "acc_norm",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "openbookqa": "acc_norm",
    "sciq": "acc_norm",
    "lambada_openai": "acc",
    "mmlu_continuation": "acc_norm",   # MMLU en version « cloze », adaptée aux petits modèles
}
BASELINES = {
    "gpt2-124M": "openai-community/gpt2",
    "gpt2-355M": "openai-community/gpt2-medium",
    "pythia-410M": "EleutherAI/pythia-410m",
    "smollm2-360M": "HuggingFaceTB/SmolLM2-360M",
}


def evaluate(path: str, tasks: list[str], limit: int | None, batch_size: str, device: str) -> dict:
    import lm_eval
    import torch
    ampere = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = "bfloat16" if ampere else "float32"
    out = lm_eval.simple_evaluate(model="hf", model_args=f"pretrained={path},dtype={dtype}", tasks=tasks,
                                  num_fewshot=0, batch_size=batch_size, limit=limit, device=device)
    scores = {}
    for task in tasks:
        res = out["results"].get(task, {})
        metric = TASKS.get(task, "acc")
        val = res.get(f"{metric},none", res.get("acc,none"))
        if val is not None:
            scores[task] = round(float(val), 4)
    return scores


def write_table(results_dir: str, tasks: list[str]) -> str:
    rows = {}
    for f in sorted(os.listdir(results_dir)):
        if f.endswith(".json"):
            rows[f[:-5]] = json.load(open(os.path.join(results_dir, f)))["scores"]
    header = "| Modèle | " + " | ".join(tasks) + " | Moyenne |\n|---|" + "---|" * (len(tasks) + 1) + "\n"
    lines = []
    for label, s in sorted(rows.items(), key=lambda kv: -_avg(kv[1], tasks)):
        cells = [f"{100 * s[t]:.1f}" if t in s else "–" for t in tasks]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | **{100 * _avg(s, tasks):.1f}** |")
    table = header + "\n".join(lines) + "\n"
    open(os.path.join(results_dir, "benchmarks.md"), "w").write(table)
    return table


def _avg(s: dict, tasks: list[str]) -> float:
    vals = [s[t] for t in tasks if t in s]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="label=chemin_ou_id_HF")
    ap.add_argument("--baselines", action="store_true", help="ajoute GPT-2, Pythia, SmolLM2")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--limit", type=int, default=None, help="exemples max par tâche (tests rapides)")
    ap.add_argument("--batch_size", default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = dict(m.split("=", 1) for m in args.models)
    if args.baselines:
        models.update(BASELINES)
    os.makedirs(args.out, exist_ok=True)
    for label, path in models.items():
        dest = os.path.join(args.out, f"{label}.json")
        if os.path.exists(dest) and not args.force:
            print(f"[{label}] déjà évalué, ignoré (--force pour refaire)")
            continue
        print(f"[{label}] évaluation de {path} …")
        scores = evaluate(path, args.tasks, args.limit, args.batch_size, args.device)
        json.dump({"path": path, "limit": args.limit, "scores": scores}, open(dest, "w"), indent=2)
        print(f"[{label}] {scores}")
    print(write_table(args.out, args.tasks))


if __name__ == "__main__":
    main()
