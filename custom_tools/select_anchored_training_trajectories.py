"""Freeze a soup-specific curriculum with only genuine successes anchored."""

import argparse
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def closest(candidates, target):
    return min(candidates, key=lambda item: (abs(item[1] - target), item[0]))


def select_object(item):
    indices = [int(value) for value in item["trajectory_indices"]]
    lifts = [float(value) for value in item[
        "diagnostic_maximum_lift_m_by_trajectory"]]
    lift_by_index = dict(zip(indices, lifts))
    successes = set(int(value) for value in item[
        "official_peak_success_source_indices"])
    successful = [(index, lift_by_index[index]) for index in sorted(successes)]
    failures = sorted(
        [(index, lift) for index, lift in lift_by_index.items()
         if index not in successes and 0.0 <= lift <= 0.30],
        key=lambda value: (value[1], value[0]))
    if not successful:
        raise ValueError("{} has no genuine BC success".format(item["object_id"]))

    roles = {}
    selected = []
    anchors = []
    if len(successful) >= 2:
        moderate = closest(successful, 0.30)
        strong = closest(
            [value for value in successful if value != moderate], 0.50)
        selected.extend((moderate, strong))
        anchors.extend((moderate[0], strong[0]))
        roles.update({
            "success_anchor_moderate": moderate[0],
            "success_anchor_strong": strong[0],
        })
        needed_failures = 2
    else:
        single = successful[0]
        selected.append(single)
        anchors.append(single[0])
        roles["success_anchor_single"] = single[0]
        needed_failures = 3
    if len(failures) < needed_failures:
        raise ValueError("{} has too few valid failures".format(item["object_id"]))

    near = failures[-1]
    remaining = failures[:-1]
    selected.append(near)
    roles["failure_near_miss"] = near[0]
    if needed_failures == 3:
        high = remaining[-1]
        selected.append(high)
        roles["failure_high"] = high[0]
        remaining = remaining[:-1]
    median = remaining[(len(remaining) - 1) // 2]
    selected.append(median)
    roles["failure_median"] = median[0]
    if len(selected) != 4 or len({value[0] for value in selected}) != 4:
        raise RuntimeError("{} did not produce four unique trajectories".format(
            item["object_id"]))
    return {
        "indices": [value[0] for value in selected],
        "anchor_indices": anchors,
        "anchor_flags": [value[0] in anchors for value in selected],
        "roles": roles,
        "maximum_lift_m": {str(index): lift for index, lift in selected},
    }


def main():
    cli = parse_cli()
    audit_path = Path(cli.audit).expanduser().resolve()
    output_path = Path(cli.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    with audit_path.open(encoding="utf-8") as handle:
        audit = yaml.safe_load(handle)
    selections = {
        item["object_id"]: select_object(item) for item in audit["objects"]}
    object_ids = [item["object_id"] for item in audit["objects"]]
    result = {
        "status": "frozen_stage1_selection",
        "stage": "soup_bc_behavior_anchored_gated_residual",
        "trajectory_root": audit["trajectory_root"],
        "source_audit": str(audit_path),
        "uses_unseen_test_objects": False,
        "selection_rule": (
            "Four trajectories per object. Use two genuine successes and two "
            "valid failures when available; if only one success exists, use one "
            "genuine anchor and three valid failures. Never label a failure as "
            "an anchor."),
        "object_ids": object_ids,
        "trajectory_indices_by_object": {
            key: value["indices"] for key, value in selections.items()},
        "anchor_indices_by_object": {
            key: value["anchor_indices"] for key, value in selections.items()},
        "anchor_flags_by_object": {
            key: value["anchor_flags"] for key, value in selections.items()},
        "selection_details": selections,
        "total_environments": 4 * len(object_ids),
        "total_anchor_environments": sum(
            len(value["anchor_indices"]) for value in selections.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, allow_unicode=True, sort_keys=False)
    print("ANCHORED_CURRICULUM=FROZEN")
    print("environments={} anchors={}".format(
        result["total_environments"], result["total_anchor_environments"]))


if __name__ == "__main__":
    main()
