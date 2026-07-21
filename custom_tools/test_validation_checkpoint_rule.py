"""CPU test for the fixed residual-PPO checkpoint selection rule."""

from custom_tools.train_residual_ppo import validation_is_better


def metrics(success, lift, failure):
    return {
        "macro_official_peak_success_rate": success,
        "macro_mean_maximum_lift_m": lift,
        "macro_failure_rate": failure,
    }


def main():
    baseline = metrics(0.10, 0.08, 0.20)
    assert validation_is_better(baseline, None)
    assert validation_is_better(metrics(0.11, 0.01, 0.90), baseline)
    assert not validation_is_better(metrics(0.09, 0.50, 0.00), baseline)
    assert validation_is_better(metrics(0.10, 0.082, 0.90), baseline)
    assert validation_is_better(metrics(0.10, 0.0805, 0.10), baseline)
    assert not validation_is_better(metrics(0.10, 0.0805, 0.30), baseline)
    print("[PASS] validation checkpoint rule: success > lift > failure")


if __name__ == "__main__":
    main()
