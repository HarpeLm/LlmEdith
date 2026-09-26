"""Génère les notebooks Kaggle/Colab. Modifie les constantes ci-dessous, puis :

    python -m notebooks.make_notebooks
"""
import json
import os

GITHUB_REPO = "https://github.com/HarpeLm/LlmEdith.git"
HF_USER = "HarpePluie"
CKPT_REPO = f"{HF_USER}/llmedith-checkpoints"   # dépôt modèle privé : checkpoints partagés
DATA_REPO = f"{HF_USER}/llmedith-data"          # dépôt dataset privé : shards tokenisés
HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def notebook(cells, accelerator=None):
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}}
    if accelerator:
        meta["accelerator"] = accelerator
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


def secrets_cell(platform):
    if platform == "kaggle":
        get = ("from kaggle_secrets import UserSecretsClient\n"
               "s = UserSecretsClient()\n"
               "os.environ['HF_TOKEN'] = s.get_secret('HF_TOKEN')\n"
               "try:\n"
               "    GH = s.get_secret('GITHUB_TOKEN')  # seulement si le dépôt GitHub est privé\n"
               "except Exception:\n"
               "    GH = None")
    else:
        get = ("from google.colab import userdata\n"
               "os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n"
               "try:\n"
               "    GH = userdata.get('GITHUB_TOKEN')  # seulement si le dépôt GitHub est privé\n"
               "except Exception:\n"
               "    GH = None")
    return code(f"""
import os, subprocess
{get}
os.environ['LLMEDITH_HUB_REPO'] = '{CKPT_REPO}'
repo = '{GITHUB_REPO}'.replace('https://', f'https://{{GH}}@') if GH else '{GITHUB_REPO}'
if not os.path.exists('LlmEdith'):
    subprocess.run(['git', 'clone', '-q', repo, 'LlmEdith'], check=True)
else:
    subprocess.run(['git', '-C', 'LlmEdith', 'pull', '-q'], check=True)
os.chdir('LlmEdith')
""")


def main():
    tpu = notebook([
        md("""
# LlmEdith : pré-entraînement JAX sur TPU (Kaggle)
**Réglages Kaggle** : *Accelerator* = **TPU VM v5e-8** (ou v3-8), *Internet* = **ON**.
*Add-ons → Secrets* : `HF_TOKEN` (jeton HF en écriture), `GITHUB_TOKEN` seulement si ton dépôt est privé.

Chaque session reprend automatiquement au dernier checkpoint (sauvegardé toutes les 30 min sur le Hub),
puis s'arrête proprement avant la limite de 9 h. Il suffit de relancer le notebook chaque semaine.
"""),
        secrets_cell("kaggle"),
        code("""
!pip install -q -e . optax
try:
    import jax
except ImportError:
    !pip install -q "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
import jax
print(jax.devices())
"""),
        code(f"""
!python -m llmedith.jax_impl.train --config configs/main.yaml --resume --max_minutes 500 \\
    --override data.hub_repo={DATA_REPO}
"""),
        md("Courbe de loss de la session :"),
        code("""
import pandas as pd
log = pd.read_csv('runs/main/log.csv', names=['step', 'split', 'loss', 'lr', 'gnorm', 'tok_s'])
ax = log[log.split == 'train'].plot(x='step', y='loss', logy=True)
log[log.split == 'val'].plot(x='step', y='loss', ax=ax, style='o-', label='val');
"""),
    ], accelerator="tpu")

    kaggle_gpu = notebook([
        md("""
# LlmEdith : PyTorch sur 2×T4 (Kaggle)
*Accelerator* = **GPU T4 x2**, *Internet* = **ON**, secrets `HF_TOKEN` / `GITHUB_TOKEN`.
Remplace un GPU local : ablations, **benchmarks** des checkpoints de la run TPU, puis **SFT/DPO**.
Les T4 n'ont pas de bf16 : l'entraînement passe automatiquement en fp16 + GradScaler.
"""),
        secrets_cell("kaggle"),
        code("!pip install -q -e '.[eval]'"),
        md("""
## A. Ablation `tiny` : Muon contre AdamW, un modèle par GPU, en parallèle
Environ 3 h. Relance la cellule dans une nouvelle session si elle est coupée : chaque run reprend.
"""),
        code(f"""
import subprocess
def run(gpu, name, opt):
    cmd = (f"CUDA_VISIBLE_DEVICES={{gpu}} python -m llmedith.torch_impl.train --config configs/tiny.yaml "
           f"--resume --max_minutes 480 --override name={{name}} train.optimizer={{opt}} data.hub_repo={DATA_REPO}")
    return subprocess.Popen(cmd + f" > {{name}}.log 2>&1", shell=True)
procs = [run(0, 'tiny_muon', 'muon'), run(1, 'tiny_adamw', 'adamw')]
for p in procs:
    p.wait()
!tail -n 3 tiny_muon.log tiny_adamw.log
"""),
        code("""
import pandas as pd
cols = ['step', 'split', 'loss', 'lr', 'gnorm', 'tok_s']
ax = None
for name in ['tiny_muon', 'tiny_adamw']:
    log = pd.read_csv(f'runs/{name}/log.csv', names=cols)
    val = log[log.split == 'val']
    ax = val.plot(x='step', y='loss', ax=ax, label=name, style='o-')
    print(name, 'val loss finale :', val.loss.iloc[-1])
"""),
        md("## B. Benchmarks du dernier checkpoint de la run principale (TPU)"),
        code(f"""
from llmedith import hub
hub.pull_folder('{CKPT_REPO}', 'main/latest', 'runs')
!python -m convert.to_hf --ckpt runs/main/latest --out exports/llmedith-main
!python -m eval.run_benchmarks --models edith-main=exports/llmedith-main --baselines --force
"""),
        md("""
## C. Alignement : SFT puis DPO (quand le pré-entraînement est terminé)
Chaque étape est envoyée sur le Hub à la fin, pour pouvoir continuer dans une autre session.
"""),
        code(f"""
from llmedith import hub
hub.pull_folder('{CKPT_REPO}', 'main/latest', 'runs')
!python -m posttrain.sft --base runs/main/latest --out runs/sft --max_examples 100000 --epochs 2
hub.push_folder('{CKPT_REPO}', 'runs/sft/latest', 'sft/latest', blocking=True)
"""),
        code(f"""
from llmedith import hub
if not os.path.exists('runs/sft/latest/meta.json'):
    hub.pull_folder('{CKPT_REPO}', 'sft/latest', 'runs')
!python -m posttrain.dpo --base runs/sft/latest --out runs/dpo --max_examples 20000
hub.push_folder('{CKPT_REPO}', 'runs/dpo/latest', 'dpo/latest', blocking=True)
"""),
    ], accelerator="nvidiaTeslaT4")

    colab = notebook([
        md("""
# LlmEdith : PyTorch sur GPU Colab
*Exécution → Modifier le type d'exécution → GPU (T4)*. Secrets 🔑 : `HF_TOKEN`, `GITHUB_TOKEN`.
Idéal pour les ablations `tiny`, le SFT et pour discuter avec le modèle.
"""),
        secrets_cell("colab"),
        code("!pip install -q -e '.[eval]'"),
        md("## Ablation `tiny` (Muon vs AdamW)"),
        code(f"""
!python -m llmedith.torch_impl.train --config configs/tiny.yaml --resume --max_minutes 170 \\
    --override data.hub_repo={DATA_REPO}
# puis : --override name=tiny_adamw train.optimizer=adamw  pour comparer
"""),
        md("## Discuter avec le modèle aligné"),
        code(f"""
from llmedith import hub
hub.pull_folder('{CKPT_REPO}', 'dpo/latest', 'runs')
!python -m convert.to_hf --ckpt runs/dpo/latest --out exports/llmedith-chat --chat
!python -m posttrain.chat exports/llmedith-chat
"""),
    ], accelerator="GPU")

    for name, nb in [("kaggle_tpu", tpu), ("kaggle_gpu", kaggle_gpu), ("colab_gpu", colab)]:
        path = os.path.join(HERE, f"{name}.ipynb")
        json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
        print("écrit", path)


if __name__ == "__main__":
    main()
