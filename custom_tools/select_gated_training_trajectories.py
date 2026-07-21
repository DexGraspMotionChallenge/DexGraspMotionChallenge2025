"""Freeze a balanced 4-trajectory curriculum from a complete BC audit."""

import argparse
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = (
    REPO_ROOT / "custom_tools/results/evaluations/"
    "bc_noise002_e100_bc_train395_isolated.yaml")
DEFAULT_OUTPUT = (
    REPO_ROOT / "custom_tools/configs/"
    "residual_full16_balanced64_trajectory_selection.yaml")


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Select two BC-success anchors and two BC failures per object.")
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def closest(candidates, target):
    return min(candidates, key=lambda item: (abs(item[1] - target), item[0]))


def select_object(item):
    indices = [int(value) for value in item["trajectory_indices"]]
    lifts = [float(value) for value in item[
        "diagnostic_maximum_lift_m_by_trajectory"]]
    if len(indices) != len(lifts):
        raise ValueError("{} index/lift lengths differ".format(item["object_id"]))
    lift_by_index = dict(zip(indices, lifts))
    successes = set(int(value) for value in item[
        "official_peak_success_source_indices"])
    successful = [(index, lift_by_index[index]) for index in sorted(successes)]
    if len(successful) < 2:
        raise ValueError("{} has fewer than two BC successes".format(
            item["object_id"]))

    moderate = closest(successful, 0.30)
    strong = closest([value for value in successful if value != moderate], 0.50)

    # Keep physically plausible failures: negative lift is a drop and lift
    # above 30 cm is likely an unstable/flying outlier rather than a near miss.
    failures = sorted(
        [(index, lift) for index, lift in lift_by_index.items()
         if index not in successes and 0.0 <= lift <= 0.30],
        key=lambda value: (value[1], value[0]))
    if len(failures) < 2:
        raise ValueError("{} has fewer than two valid BC failures".format(
            item["object_id"]))
    near_miss = failures[-1]
    remaining = [value for value in failures if value != near_miss]
    median_failure = remaining[(len(remaining) - 1) // 2]
    selected = [moderate, strong, near_miss, median_failure]
    return {
        "indices": [value[0] for value in selected],
        "roles": {
            "success_anchor_moderate": moderate[0],
            "success_anchor_strong": strong[0],
            "failure_near_miss": near_miss[0],
            "failure_median": median_failure[0],
        },
        "maximum_lift_m": {str(index): lift for index, lift in selected},
    }


def main():
    cli = parse_cli()
    audit_path = Path(cli.audit).expanduser().resolve()
    output_path = Path(cli.output).expanduser().resolve()
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = yaml.safe_load(handle)
    objects = [item["object_id"] for item in audit["objects"]]
    selections = {item["object_id"]: select_object(item)
                  for item in audit["objects"]}
    result = {
        "status": "frozen_stage1_selection",
        "stage": "noise_bc_balanced64_gated_residual_comparison",
        "trajectory_root": audit["trajectory_root"],
        "source_audit": str(audit_path),
        "uses_unseen_test_objects": False,
        "selection_rule": (
            "Per object: two successful BC anchors nearest 0.30 m and 0.50 m "
            "lift; one highest-lift valid failure and one median valid failure. "
            "Valid failures have lift in [0, 0.30] m."),
        "warning": (
            "Some former fixed indices 0-9 are now training data; use the "
            "separate held-out validation set for model comparison."),
        "object_ids": objects,
        "trajectory_indices_by_object": {
            object_id: selections[object_id]["indices"] for object_id in objects},
        "selection_details": selections,
        "total_environments": 4 * len(objects),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, allow_unicode=True, sort_keys=False)
    print("[PASS] selected {} environments across {} objects: {}".format(
        result["total_environments"], len(objects), output_path))


if __name__ == "__main__":
    main()
