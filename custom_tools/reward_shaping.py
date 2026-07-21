"""Pure reward-shaping helpers shared by residual training and CPU tests."""

import torch


LIFT_PROGRESS_MODES = ("signed_delta", "max_height_progress")


def compute_lift_progress(height, previous_height, maximum_height,
                          mode="signed_delta", clamp_m=0.03):
    """Return per-step lift progress and the updated episode maximum height."""
    if mode not in LIFT_PROGRESS_MODES:
        raise ValueError(
            "lift_progress_mode must be one of {}, got {!r}".format(
                LIFT_PROGRESS_MODES, mode))
    if mode == "signed_delta":
        progress = (height - previous_height).clamp(-clamp_m, clamp_m)
    else:
        # Reward only a new episode height record.  A downward movement earns
        # zero here, and returning to an old height cannot farm reward.
        progress = (height - maximum_height).clamp(0.0, clamp_m)
    updated_maximum = torch.maximum(maximum_height, height)
    return progress, updated_maximum
