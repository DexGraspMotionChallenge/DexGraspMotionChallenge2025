"""CPU test for category-balanced PPO advantage normalization."""

import torch

from custom_tools.residual_ppo import normalize_advantages


def main():
    torch.manual_seed(11)
    samples = 128
    # Three categories carry real signals at very different scales.  The last
    # has tiny noise and must not be amplified to unit variance.
    raw = torch.cat((
        2.0 + 2.0 * torch.randn(samples),
        -10.0 + 10.0 * torch.randn(samples),
        30.0 + 30.0 * torch.randn(samples),
        0.001 * torch.randn(samples),
    ))
    groups = torch.arange(4).repeat_interleave(samples)
    normalized, stats = normalize_advantages(
        raw, mode="category", group_ids=groups,
        min_std=1.0, clip_value=5.0)
    assert torch.isfinite(normalized).all()
    for group in (0, 1, 2):
        assert abs(stats[group]["normalized_mean"]) < 1e-5
        assert 0.95 < stats[group]["normalized_std"] < 1.05
    assert stats[3]["divisor"] == 1.0
    assert stats[3]["normalized_std"] < 0.01
    assert normalized.abs().max() <= 5.0
    print("[PASS] category advantage normalization")
    for group, group_stats in stats.items():
        print("group{} raw_std={:.4f} divisor={:.4f} normalized_std={:.4f}".format(
            group, group_stats["raw_std"], group_stats["divisor"],
            group_stats["normalized_std"]))


if __name__ == "__main__":
    main()
