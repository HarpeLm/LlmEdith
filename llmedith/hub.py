"""Synchronisation des checkpoints avec le Hugging Face Hub (dépôt privé).

C'est le stockage commun à Kaggle, Colab et au PC local. Une session peut
reprendre là où une autre s'est arrêtée.
"""
from __future__ import annotations

import os
import threading

_upload_thread: threading.Thread | None = None


def push_folder(repo: str, local_dir: str, path_in_repo: str, blocking: bool = False) -> None:
    """Envoie un dossier sur le Hub, en arrière-plan par défaut, pour ne pas bloquer l'entraînement."""
    global _upload_thread
    if not repo:
        return
    from huggingface_hub import HfApi

    def _run():
        try:
            api = HfApi()
            api.create_repo(repo, private=True, exist_ok=True)
            api.upload_folder(repo_id=repo, folder_path=local_dir, path_in_repo=path_in_repo,
                              commit_message=f"checkpoint {path_in_repo}")
            print(f"[hub] envoyé {local_dir} -> {repo}/{path_in_repo}")
        except Exception as e:  # une coupure réseau ne doit pas tuer l'entraînement
            print(f"[hub] échec de l'envoi : {e}")

    wait_uploads()
    if blocking:
        _run()
    else:
        _upload_thread = threading.Thread(target=_run, daemon=True)
        _upload_thread.start()


def wait_uploads() -> None:
    if _upload_thread is not None and _upload_thread.is_alive():
        _upload_thread.join()


def pull_folder(repo: str, path_in_repo: str, local_root: str) -> str | None:
    """Télécharge `path_in_repo` depuis le Hub. Renvoie le chemin local, ou None s'il n'existe pas."""
    if not repo:
        return None
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo_id=repo, allow_patterns=[f"{path_in_repo}/*"], local_dir=local_root)
    except Exception as e:
        print(f"[hub] rien à reprendre ({e.__class__.__name__})")
        return None
    path = os.path.join(local_root, path_in_repo)
    return path if os.path.exists(os.path.join(path, "meta.json")) else None
