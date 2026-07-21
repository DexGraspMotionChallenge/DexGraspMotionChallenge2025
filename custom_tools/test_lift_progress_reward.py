"""CPU tests for signed and maximum-height lift-progress rewards."""

import torch

from custom_tools.reward_shaping import compute_lift_progress


def rollout(heights, mode):
    previous = torch.tensor([heights[0]], dtype=torch.float32)
    maximum = previous.clone()
    rewards = []
    for value in heights[1:]:
        height = torch.tensor([value], dtype=torch.float32)
        progress, maximum = compute_lift_progress(
            height, previous, maximum, mode=mode)
        rewards.append(float(progress.item()))
        previous = height
    return rewards, float(maximum.item())


def main():
    # Rise 1 cm, dip 0.4 cm, return below the old maximum, then set a 2 cm max.
    heights = [0.00, 0.01, 0.006, 0.009, 0.02]
    signed, signed_max = rollout(heights, "signed_delta")
    record, record_max = rollout(heights, "max_height_progress")

    expected_signed = [0.01, -0.004, 0.003, 0.011]
    expected_record = [0.01, 0.0, 0.0, 0.01]
    assert torch.allclose(torch.tensor(signed), torch.tensor(expected_signed))
    assert torch.allclose(torch.tensor(record), torch.tensor(expected_record))
    assert abs(signed_max - 0.02) < 1e-6
    assert abs(record_max - 0.02) < 1e-6
    assert abs(sum(record) - 0.02) < 1e-6

    try:
        rollout(heights, "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown lift-progress mode was accepted")

    print("[PASS] lift progress reward: signed={} max_height={}".format(
        signed, record))


if __name__ == "__main__":
    main()
