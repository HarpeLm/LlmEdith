"""Planning de learning rate WSD (Warmup-Stable-Decay).

Cooldown en « 1 - sqrt » : meilleur que linéaire ou cosinus d'après
Hägele et al. 2024, « Scaling Laws and Compute-Optimal Training
Beyond Fixed Training Durations ».
L'avantage du WSD : on peut prolonger l'entraînement tant qu'on est sur
le plateau, et décider plus tard quand lancer la décroissance.
"""
import math


def wsd_lr_factor(step: int, total_steps: int, warmup_steps: int, decay_frac: float,
                  min_frac: float = 0.0) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, int(total_steps * decay_frac))
    decay_start = total_steps - decay_steps
    if step < decay_start:
        return 1.0
    progress = min(1.0, (step - decay_start) / decay_steps)
    return min_frac + (1.0 - min_frac) * (1.0 - math.sqrt(progress))


def in_decay_phase(step: int, total_steps: int, decay_frac: float) -> bool:
    return step >= total_steps - max(1, int(total_steps * decay_frac))
