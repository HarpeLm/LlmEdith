# LlmEdith

Un LLM anglais **construit entièrement de zéro** : tokenizer, pré-entraînement, alignement et benchmarks.
Il est entraîné uniquement sur du calcul gratuit : **TPU Kaggle (JAX)**, **GPU Kaggle et Colab (PyTorch)** et une **RTX 4060 Ti**.

Objectif : un modèle de **~320M paramètres** entraîné sur **~20 milliards de tokens**, qui bat GPT-2 (355M) et Pythia-410M
et approche SmolLM2-360M.

## Architecture

Transformer décodeur « moderne », sur le modèle de Llama 3, Qwen 3 et OLMo 2 :

| Élément | Choix | Source |
|---|---|---|
| Normalisation | RMSNorm en pre-norm + **QK-norm** | OLMo 2, Qwen 3 : stabilité à LR élevé |
| Attention | **GQA** (16 têtes de requêtes, 4 têtes KV), RoPE θ = 10 000 | Llama 2/3 |
| MLP | **SwiGLU** | Shazeer 2020 |
| Divers | Pas de biais, embeddings liés, z-loss 1e-4 | PaLM, SmolLM |
| Optimiseur | **Muon** (matrices) + AdamW (embeddings et normes) | modded-nanogpt, Moonlight (2025) |
| Planning du LR | **WSD** avec cooldown 1-√x, puis *annealing* sur des données de haute qualité | MiniCPM, Hägele et al. 2024 |
| Tokenizer | BPE byte-level, 32 768 tokens, chiffres isolés | GPT-4 / SmolLM |
| Données | FineWeb-Edu 60 %, DCLM 25 %, code 8 %, FineMath 7 % | FineWeb-Edu, DCLM, SmolLM2 |

Les poids ont **exactement le format de `Qwen3ForCausalLM`**. Un export marche donc directement avec
`transformers`, `lm-eval-harness`, vLLM et llama.cpp. Les versions PyTorch et JAX sont **identiques à 1e-4 près**
(voir les tests), et un checkpoint de l'une se reprend dans l'autre.

| Config | Paramètres | Usage |
|---|---|---|
| `configs/tiny.yaml` | 22M | Ablations sur la 4060 Ti (~1-2 h pour 500M tokens) |
| `configs/small.yaml` | 101M | Étape intermédiaire, comparable à GPT-2 124M |
| `configs/main.yaml` | 323M | Run principale sur TPU (20 Md de tokens, ~38k steps de 0,5M tokens) |

## Répartition du calcul gratuit

| Ressource | Rôle | Ordre de grandeur |
|---|---|---|
| **TPU Kaggle v5e-8 / v3-8** (~20 h/semaine) | Pré-entraînement principal (JAX) | ~6 à 18 Md de tokens/semaine pour 323M |
| **RTX 4060 Ti** | Tokenizer et données (CPU), ablations `tiny`, SFT, DPO, chat | illimité |
| **Kaggle 2×T4** (~30 h/semaine) | Ablations `small` (torchrun), benchmarks des checkpoints | fp16 |
| **Colab T4** | Ablations, SFT, démo | sessions courtes |

Tout est **reprenable** :
- checkpoints toutes les 30 min sur un dépôt privé du HF Hub ;
- data loader déterministe ;
- `--max_minutes` pour s'arrêter proprement avant la fin du quota.

## Mode d'emploi

### 0. Installation (PC local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch,jax,data,eval,dev]"
huggingface-cli login          # jeton HF en écriture
pytest                          # 11 tests : parité PyTorch/HF/JAX, sur-apprentissage, data loader…
```

### 1. Tokenizer (~30 min sur CPU)
```bash
python -m tokenizer.train_tokenizer --num_bytes 5e9
git add tokenizer/llmedith-32k/tokenizer.json   # il est versionné, les notebooks le récupèrent via git
```

### 2. Données (sur le PC, reprenable, plusieurs heures)
```bash
python -m data.prepare --mix train  --total_tokens 25e9 --out shards
python -m data.prepare --mix anneal --total_tokens 3e9  --out shards --no_val
python -m data.prepare --upload TON_PSEUDO_HF/llmedith-data --out shards
```
Il faut ~55 Go de disque. Pour démarrer plus vite, tu peux commencer avec `--total_tokens 5e9`, puis relancer plus tard : la préparation reprend là où elle s'est arrêtée.

### 3. Ablations locales (4060 Ti)
```bash
python -m llmedith.torch_impl.train --config configs/tiny.yaml
python -m llmedith.torch_impl.train --config configs/tiny.yaml --override name=tiny_adamw train.optimizer=adamw
```
Compare les `val loss` dans `runs/*/log.csv`. Muon devrait l'emporter.

### 4. Run principale (TPU Kaggle)
1. Pousse ce dépôt sur GitHub.
2. Modifie les constantes en haut de `notebooks/make_notebooks.py`, puis lance `python -m notebooks.make_notebooks`.
3. Importe `notebooks/kaggle_tpu.ipynb` dans Kaggle, avec *Accelerator* = TPU VM et les secrets `HF_TOKEN` et `GITHUB_TOKEN`.
4. Lance-le chaque semaine : il reprend tout seul.

### 5. Benchmarks
```bash
python -m convert.to_hf --ckpt runs/main/latest --out exports/llmedith-main
python -m eval.run_benchmarks --models edith-main=exports/llmedith-main --baselines
```
Le tableau est généré dans `results/benchmarks.md`. Tâches en 0-shot :
- HellaSwag, ARC-Easy, ARC-Challenge, PIQA, WinoGrande, OpenBookQA, SciQ, LAMBADA ;
- MMLU (cloze).

Modèles de référence : GPT-2 124M/355M, Pythia-410M, SmolLM2-360M.
Astuce : évalue un checkpoint par semaine pour tracer la courbe « score en fonction des tokens vus ».

### 6. Alignement (4060 Ti)
```bash
python -m posttrain.sft --base runs/main/latest --out runs/sft
python -m posttrain.dpo --base runs/sft/latest --out runs/dpo
python -m convert.to_hf --ckpt runs/dpo/latest --out exports/llmedith-chat --chat
python -m posttrain.chat exports/llmedith-chat
```

## Structure
```
llmedith/            cœur partagé : config, data loader, planning WSD, conversion, sync Hub
  torch_impl/        modèle, Muon et boucle d'entraînement PyTorch (AMP, DDP, reprise)
  jax_impl/          modèle, Muon et boucle d'entraînement JAX (scan, remat, sharding TPU)
tokenizer/           entraînement du tokenizer BPE
data/                téléchargement, mélange et tokenisation en shards uint16
convert/             export au format Hugging Face (Qwen3)
eval/                benchmarks lm-eval-harness et modèles de référence
posttrain/           SFT (smoltalk), DPO (UltraFeedback), chat
notebooks/           Kaggle TPU, Kaggle 2×T4, Colab
configs/             tailles de modèle et mélange de données
tests/               tests de cohérence
```
