# curriculum_sampler.py
def compute_crop_size(epoch, max_epoch, min_size=256, max_size=1024):
    """
    Patch‑wise curriculum
    Early epochs: train only on random 256×256 crops that happen to be mostly single material (≥80% confidence from SegFormer).
    Later epochs: gradually increase crop size to expose the model to mixed‑material context (curriculum). The schedule can be automated:

        Linearly increase crop size with epoch.
    """
    alpha = epoch / max_epoch
    return int(min_size + alpha * (max_size - min_size))
