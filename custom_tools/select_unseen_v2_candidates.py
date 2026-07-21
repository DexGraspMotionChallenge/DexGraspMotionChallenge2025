"""Select policy-blind unseen-v2 candidates from previously unused objects."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from custom_tools.select_object_split import (
    FEATURE_KEYS, collect_category, normalized_features)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-manifest", default=str(
        REPO_ROOT / "custom_tools/configs/object_split_candidates.json"))
    parser.add_argument("--dataset-root", default=str(
        REPO_ROOT / "external_data/dataset"))
    parser.add_argument("--mesh-root", default=str(
        REPO_ROOT / "external_data/meshdata"))
    parser.add_argument("--output-json", default=str(
        REPO_ROOT / "custom_tools/configs/unseen_v2_candidates.json"))
    parser.add_argument("--output-csv", default=str(
        REPO_ROOT / "custom_tools/configs/unseen_v2_candidates.csv"))
    parser.add_argument("--min-trajectories", type=int, default=40)
    return parser.parse_args()


def select_four(objects, excluded):
    features = normalized_features(objects)
    center = np.median(features, axis=0)
    center_distance = np.linalg.norm(features - center, axis=1)
    available = [index for index, item in enumerate(objects)
                 if item["object_id"] not in excluded]
    if len(available) < 4:
        raise RuntimeError("fewer than four unused eligible objects")

    # First choose a representative object near the category center.  The
    # second is geometrically different but not in the most extreme 10%,
    # avoiding a hand-picked easy or pathological final set.
    first = min(available, key=lambda i: (
        center_distance[i], objects[i]["object_id"]))
    cutoff = float(np.percentile(center_distance[available], 90))
    moderate = [i for i in available if i != first and center_distance[i] <= cutoff]
    second = max(moderate, key=lambda i: (
        float(np.linalg.norm(features[i] - features[first])),
        objects[i]["object_id"]))
    chosen = [first, second]

    # Two backups maximize distance from the already selected candidates.
    while len(chosen) < 4:
        remaining = [i for i in available if i not in chosen]
        next_index = max(remaining, key=lambda i: (
            min(float(np.linalg.norm(features[i] - features[j])) for j in chosen),
            -center_distance[i], objects[i]["object_id"]))
        chosen.append(next_index)

    low, high = np.percentile(center_distance, [33, 66])
    selected = []
    for rank, index in enumerate(chosen, 1):
        item = dict(objects[index])
        item["geometry_distance_to_category_center"] = float(center_distance[index])
        item["geometry_proxy"] = (
            "typical" if center_distance[index] <= low else
            "medium" if center_distance[index] <= high else "unusual")
        item["candidate_rank"] = rank
        selected.append(item)
    return selected


def main():
    cli = parse_cli()
    with Path(cli.previous_manifest).open(encoding="utf-8") as handle:
        previous = json.load(handle)
    excluded = set(previous["objects"])
    categories = list(previous["criteria"]["categories"])
    manifest = {
        "status": "policy_blind_candidates_not_yet_replayed",
        "warning": "Do not evaluate a learned policy before this split is frozen.",
        "selection_rule": (
            "Exclude all objects in the previous 32-object candidate manifest; "
            "choose one category-center object, one diverse non-extreme object, "
            "and two farthest-point backups using geometry only."),
        "criteria": {
            "categories": categories,
            "test_per_category": 2,
            "backups_per_category": 2,
            "min_trajectories": cli.min_trajectories,
            "feature_keys": list(FEATURE_KEYS),
        },
        "categories": {},
        "objects": {},
        "excluded_previous_object_count": len(excluded),
    }
    rows = []
    for category in categories:
        objects, _ = collect_category(
            category, Path(cli.dataset_root), Path(cli.mesh_root),
            cli.min_trajectories, 0.01, 0.01)
        chosen = select_four(objects, excluded)
        test = [item["object_id"] for item in chosen[:2]]
        backups = [item["object_id"] for item in chosen[2:]]
        manifest["categories"][category] = {
            "eligible_count": len(objects), "train": [],
            "test": test, "backups": backups}
        for item in chosen:
            item["category"] = category
            item["split"] = "test" if item["candidate_rank"] <= 2 else "backups"
            manifest["objects"][item["object_id"]] = item
            rows.append(item)
        print("{}: test={}, backups={}".format(category, test, backups))

    output_json = Path(cli.output_json).resolve()
    output_csv = Path(cli.output_csv).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    fields = ("category", "split", "candidate_rank", "object_id",
              "trajectory_count", "geometry_proxy",
              "geometry_distance_to_category_center",
              "physical_longest_extent", "physical_bbox_volume",
              "bbox_aspect_ratio", "convex_piece_count", "vertex_count",
              "rotation_angle_std", "final_action_dispersion")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    print("UNSEEN_V2_CANDIDATES=READY")


if __name__ == "__main__":
    main()
