"""Select a reproducible easy/near-success/medium Stage-1 curriculum."""

import argparse
import math
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Select four Stage-1 trajectories per representative object.")
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--output", default=str(
        REPO_ROOT / "custom_tools/configs/residual_stage1_trajectory_selection.yaml"))
    return parser.parse_args()


def pair_distance(rotations, scales, first, second):
    relative = rotations[first] @ rotations[second].T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    rotation_distance = math.acos(cosine) / math.pi
    scale_range = max(float(scales.max() - scales.min()), 1e-6)
    scale_distance = abs(float(scales[first] - scales[second])) / scale_range
    return rotation_distance + 0.5 * scale_distance


def diversity(rotations, scales, candidate, selected):
    if not selected:
        return 0.0
    return min(pair_distance(rotations, scales, candidate, item)
               for item in selected)


def choose_near_failure(failures, lifts, rotations, scales, selected):
    remaining = [index for index in failures if index not in selected]
    top_count = min(len(remaining), max(6, math.ceil(len(failures) * 0.25)))
    pool = sorted(remaining, key=lambda index: (-lifts[index], index))[:top_count]
    pool_lifts = np.asarray([lifts[index] for index in pool])
    lift_min, lift_max = float(pool_lifts.min()), float(pool_lifts.max())
    denominator = max(lift_max - lift_min, 1e-8)
    return max(
        pool,
        key=lambda index: (
            0.7 * (float(lifts[index]) - lift_min) / denominator
            + 0.3 * diversity(rotations, scales, index, selected),
            -index,
        ),
    )


def choose_medium_failure(failures, lifts, rotations, scales, selected):
    remaining = [index for index in failures if index not in selected]
    median_lift = float(np.median([lifts[index] for index in failures]))
    pool = sorted(
        remaining, key=lambda index: (abs(float(lifts[index]) - median_lift), index)
    )[:min(6, len(remaining))]
    return max(
        pool,
        key=lambda index: (diversity(rotations, scales, index, selected), -index),
    )


def select_object(result, trajectory_root):
    object_id = result["object_id"]
    source = np.load(
        trajectory_root / "{}.npy".format(object_id), allow_pickle=True).item()
    rotations = np.asarray(source["obj_rotmat"])
    scales = np.asarray(source["obj_scale"])
    lifts = np.asarray(result["diagnostic_maximum_lift_m_by_trajectory"])
    if not (len(rotations) == len(scales) == len(lifts)):
        raise ValueError("Trajectory count mismatch for {}".format(object_id))
    successes = sorted(result["diagnostic_ever_success_indices"])
    success_set = set(successes)
    failures = [index for index in range(len(lifts)) if index not in success_set]
    if len(failures) < 3:
        raise ValueError("Not enough failed trajectories for {}".format(object_id))

    if successes:
        anchor = min(successes, key=lambda index: (abs(lifts[index] - 0.30), index))
        anchor_kind = "single_object_success_anchor"
    else:
        anchor = max(range(len(lifts)), key=lambda index: (lifts[index], -index))
        anchor_kind = "highest_lift_failure_anchor"
    selected = [anchor]
    roles = [anchor_kind]
    while len(selected) < 3:
        selected.append(choose_near_failure(
            failures, lifts, rotations, scales, selected))
        roles.append("high_lift_failure")
    selected.append(choose_medium_failure(
        failures, lifts, rotations, scales, selected))
    roles.append("medium_failure")

    return {
        "object_id": object_id,
        "trajectory_indices": selected,
        "selected": [
            {
                "index": int(index),
                "role": role,
                "diagnostic_ever_success": bool(index in success_set),
                "diagnostic_maximum_lift_m": float(lifts[index]),
                "object_scale": float(scales[index]),
            }
            for index, role in zip(selected, roles)
        ],
    }


def main(cli):
    evaluation_path = Path(cli.evaluation).expanduser().resolve()
    trajectory_root = Path(cli.trajectory_root).expanduser().resolve()
    output_path = Path(cli.output).expanduser().resolve()
    with evaluation_path.open("r", encoding="utf-8") as handle:
        evaluation = yaml.safe_load(handle)
    if Path(evaluation["trajectory_root"]).resolve() != trajectory_root:
        raise ValueError("Evaluation and requested trajectory roots differ")
    objects = [select_object(result, trajectory_root)
               for result in evaluation["objects"]]
    output = {
        "status": "candidate_stage1_selection",
        "source_evaluation": str(evaluation_path),
        "trajectory_root": str(trajectory_root),
        "uses_unseen_test_objects": False,
        "selection_rule": (
            "One single-object-evaluation success closest to 30 cm (or highest-lift "
            "failure), two "
            "high-lift failed trajectories "
            "with rotation/scale diversity, and one diverse median-lift failure."),
        "object_ids": [item["object_id"] for item in objects],
        "trajectory_indices_by_object": {
            item["object_id"]: item["trajectory_indices"] for item in objects},
        "objects": objects,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, allow_unicode=True, sort_keys=False)
    for item in objects:
        print("{}: {}".format(item["object_id"], item["trajectory_indices"]))
    print("Saved Stage-1 selection: {}".format(output_path))


if __name__ == "__main__":
    main(parse_cli())
