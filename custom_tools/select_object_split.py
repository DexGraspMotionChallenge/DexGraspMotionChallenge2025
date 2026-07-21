"""Build a deterministic, geometry-diverse DexGrasp object split.

This tool is deliberately CPU-only.  It filters raw GraspM3 objects using
trajectory and mesh integrity, then proposes train/test/back-up objects from
each requested ShapeNet ``core`` category.  The difficulty labels are geometry
proxies only; actual replay success must be checked before the split is frozen.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select a reproducible, geometry-diverse object split.")
    parser.add_argument(
        "--categories", nargs="+", default=["bottle", "mug", "bowl", "camera"])
    parser.add_argument("--train-per-category", type=int, default=4)
    parser.add_argument("--test-per-category", type=int, default=1)
    parser.add_argument("--backups-per-category", type=int, default=3)
    parser.add_argument("--min-trajectories", type=int, default=20)
    parser.add_argument("--min-rotation-std", type=float, default=0.01)
    parser.add_argument("--min-action-dispersion", type=float, default=0.01)
    parser.add_argument(
        "--dataset-root", default=str(REPO_ROOT / "external_data" / "dataset"))
    parser.add_argument(
        "--mesh-root", default=str(REPO_ROOT / "external_data" / "meshdata"))
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "custom_tools" / "configs" /
                    "object_split_candidates.json"))
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "custom_tools" / "configs" /
                    "object_split_candidates.csv"))
    return parser.parse_args()


def mesh_metrics(mesh_root, object_id):
    coacd_dir = mesh_root / object_id / "coacd"
    obj_path = coacd_dir / "decomposed.obj"
    urdf_path = coacd_dir / "coacd_1.urdf"
    convex_paths = sorted(coacd_dir.glob("coacd_convex_piece_*.obj"))
    if not obj_path.is_file() or not urdf_path.is_file() or not convex_paths:
        raise ValueError("incomplete mesh")

    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    vertex_count = 0
    face_count = 0
    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()
                if len(values) >= 4:
                    vertex = np.asarray(values[1:4], dtype=np.float64)
                    minimum = np.minimum(minimum, vertex)
                    maximum = np.maximum(maximum, vertex)
                    vertex_count += 1
            elif line.startswith("f "):
                face_count += 1
    if vertex_count == 0 or not np.isfinite(minimum).all():
        raise ValueError("mesh has no finite vertices")
    extents = np.maximum(maximum - minimum, 1e-8)
    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "convex_piece_count": len(convex_paths),
        "bbox_x": float(extents[0]),
        "bbox_y": float(extents[1]),
        "bbox_z": float(extents[2]),
        "bbox_aspect_ratio": float(extents.max() / extents.min()),
        "bbox_volume": float(np.prod(extents)),
    }


def trajectory_metrics(path):
    data = np.load(str(path), allow_pickle=True).item()
    required = ("grasp_seqs", "obj_rotmat", "obj_scale")
    if any(key not in data for key in required):
        raise ValueError("missing trajectory keys")
    grasp_seqs = np.asarray(data["grasp_seqs"])
    rotations = np.asarray(data["obj_rotmat"])
    scales = np.asarray(data["obj_scale"])
    if (grasp_seqs.ndim != 3 or grasp_seqs.shape[1:] != (70, 28)
            or rotations.shape != (len(grasp_seqs), 3, 3)
            or scales.shape != (len(grasp_seqs),)):
        raise ValueError("unexpected trajectory shape")
    if not all(np.isfinite(array).all() for array in (grasp_seqs, rotations, scales)):
        raise ValueError("non-finite trajectory values")

    traces = np.trace(rotations, axis1=1, axis2=2)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    final_actions = grasp_seqs[:, -1, :]
    return {
        "trajectory_count": int(len(grasp_seqs)),
        "scale_min": float(scales.min()),
        "scale_median": float(np.median(scales)),
        "scale_max": float(scales.max()),
        "rotation_angle_std": float(np.std(angles)),
        "final_action_dispersion": float(np.mean(np.std(final_actions, axis=0))),
    }


def collect_category(
        category, dataset_root, mesh_root, min_trajectories,
        min_rotation_std, min_action_dispersion):
    objects = []
    rejected = []
    for path in sorted(dataset_root.glob("core-{}-*.npy".format(category))):
        object_id = path.stem
        try:
            metrics = trajectory_metrics(path)
            if metrics["trajectory_count"] < min_trajectories:
                raise ValueError("too few trajectories")
            if metrics["rotation_angle_std"] < min_rotation_std:
                raise ValueError("near-duplicate object rotations")
            if metrics["final_action_dispersion"] < min_action_dispersion:
                raise ValueError("near-duplicate final grasp actions")
            metrics.update(mesh_metrics(mesh_root, object_id))
            scale = metrics["scale_median"]
            metrics["physical_longest_extent"] = float(
                max(metrics["bbox_x"], metrics["bbox_y"], metrics["bbox_z"]) * scale)
            metrics["physical_bbox_volume"] = float(metrics["bbox_volume"] * scale ** 3)
            metrics["object_id"] = object_id
            metrics["category"] = category
            objects.append(metrics)
        except (OSError, ValueError, KeyError) as error:
            rejected.append({"object_id": object_id, "reason": str(error)})
    return objects, rejected


FEATURE_KEYS = (
    "bbox_aspect_ratio",
    "physical_longest_extent",
    "physical_bbox_volume",
    "convex_piece_count",
    "vertex_count",
    "rotation_angle_std",
    "final_action_dispersion",
)


def normalized_features(objects):
    raw = np.asarray([
        [
            math.log1p(item[key]) if key in {
                "bbox_aspect_ratio", "physical_bbox_volume",
                "convex_piece_count", "vertex_count"
            } else item[key]
            for key in FEATURE_KEYS
        ]
        for item in objects
    ], dtype=np.float64)
    low = np.percentile(raw, 5, axis=0)
    high = np.percentile(raw, 95, axis=0)
    return np.clip((raw - low) / np.maximum(high - low, 1e-8), 0.0, 1.0)


def pairwise_distances(features):
    delta = features[:, None, :] - features[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def select_category(objects, train_count, test_count, backup_count):
    required = train_count + test_count + backup_count
    if len(objects) < required:
        raise RuntimeError(
            "only {} eligible objects, but {} are required".format(len(objects), required))

    features = normalized_features(objects)
    distances = pairwise_distances(features)
    center = np.median(features, axis=0)
    center_distance = np.linalg.norm(features - center, axis=1)
    trajectory_counts = np.asarray([item["trajectory_count"] for item in objects])
    low_cut, high_cut = np.percentile(center_distance, [33, 66])
    strata = {
        "typical": [index for index in range(len(objects))
                    if center_distance[index] <= low_cut],
        "medium": [index for index in range(len(objects))
                   if low_cut < center_distance[index] <= high_cut],
        "unusual": [index for index in range(len(objects))
                    if center_distance[index] > high_cut],
    }
    for name, indices in strata.items():
        if not indices:
            raise RuntimeError("empty geometry stratum: {}".format(name))

    typical = min(
        strata["typical"],
        key=lambda index: (
            center_distance[index] - 0.02 * trajectory_counts[index],
            objects[index]["object_id"],
        ),
    )
    train_indices = [int(typical)]

    medium_target = max(0, train_count - 2)
    for _ in range(medium_target):
        remaining_medium = [
            index for index in strata["medium"] if index not in train_indices]
        next_index = max(
            remaining_medium,
            key=lambda index: (
                float(np.min(distances[index, train_indices]))
                + 0.05 * trajectory_counts[index] / trajectory_counts.max(),
                objects[index]["object_id"],
            ),
        )
        train_indices.append(int(next_index))

    if len(train_indices) < train_count:
        unusual = max(
            strata["unusual"],
            key=lambda index: (
                float(np.min(distances[index, train_indices]))
                + 0.05 * trajectory_counts[index] / trajectory_counts.max(),
                objects[index]["object_id"],
            ),
        )
        train_indices.append(int(unusual))

    while len(train_indices) < train_count:
        remaining = [index for index in range(len(objects)) if index not in train_indices]
        next_index = max(
            remaining,
            key=lambda index: (
                float(np.min(distances[index, train_indices])),
                trajectory_counts[index],
                objects[index]["object_id"],
            ),
        )
        train_indices.append(int(next_index))

    remaining = [index for index in range(len(objects)) if index not in train_indices]
    distance_to_train = np.asarray([
        float(np.min(distances[index, train_indices])) for index in remaining])
    target_distance = float(np.percentile(distance_to_train, 75))
    ranked_test = sorted(
        remaining,
        key=lambda index: (
            abs(float(np.min(distances[index, train_indices])) - target_distance),
            -trajectory_counts[index],
            objects[index]["object_id"],
        ),
    )
    test_indices = [int(index) for index in ranked_test[:test_count]]

    chosen = train_indices + test_indices
    backup_indices = []
    for stratum_name in ("typical", "medium", "unusual"):
        if len(backup_indices) >= backup_count:
            break
        remaining = [
            index for index in strata[stratum_name] if index not in chosen]
        if not remaining:
            continue
        next_index = max(
            remaining,
            key=lambda index: (
                trajectory_counts[index],
                float(np.min(distances[index, chosen])),
                objects[index]["object_id"],
            ),
        )
        backup_indices.append(int(next_index))
        chosen.append(int(next_index))
    while len(backup_indices) < backup_count:
        remaining = [index for index in range(len(objects)) if index not in chosen]
        next_index = max(
            remaining,
            key=lambda index: (
                trajectory_counts[index],
                float(np.min(distances[index, chosen])),
                objects[index]["object_id"],
            ),
        )
        backup_indices.append(int(next_index))
        chosen.append(int(next_index))

    for index, item in enumerate(objects):
        if center_distance[index] <= low_cut:
            proxy = "typical"
        elif center_distance[index] <= high_cut:
            proxy = "medium"
        else:
            proxy = "unusual"
        item["geometry_distance_to_category_center"] = float(center_distance[index])
        item["geometry_proxy"] = proxy

    return {
        "train": [objects[index]["object_id"] for index in train_indices],
        "test": [objects[index]["object_id"] for index in test_indices],
        "backups": [objects[index]["object_id"] for index in backup_indices],
    }


def write_csv(path, selected_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category", "split", "rank", "object_id", "trajectory_count",
        "geometry_proxy", "geometry_distance_to_category_center",
        "scale_median", "physical_longest_extent", "physical_bbox_volume",
        "bbox_aspect_ratio", "convex_piece_count", "vertex_count",
        "rotation_angle_std", "final_action_dispersion",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    args = parse_args()
    if min(args.train_per_category, args.test_per_category,
           args.backups_per_category, args.min_trajectories) < 1:
        raise ValueError("all counts must be positive")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    mesh_root = Path(args.mesh_root).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    manifest = {
        "selection_status": "geometry_candidates_only",
        "warning": (
            "geometry_proxy is not grasp difficulty; replay and preprocessing "
            "must pass before this split is frozen"),
        "criteria": {
            "categories": args.categories,
            "train_per_category": args.train_per_category,
            "test_per_category": args.test_per_category,
            "backups_per_category": args.backups_per_category,
            "min_trajectories": args.min_trajectories,
            "min_rotation_std": args.min_rotation_std,
            "min_action_dispersion": args.min_action_dispersion,
            "feature_keys": list(FEATURE_KEYS),
        },
        "categories": {},
        "objects": {},
        "rejected_counts": {},
    }
    csv_rows = []

    for category in args.categories:
        objects, rejected = collect_category(
            category, dataset_root, mesh_root, args.min_trajectories,
            args.min_rotation_std, args.min_action_dispersion)
        split = select_category(
            objects, args.train_per_category, args.test_per_category,
            args.backups_per_category)
        manifest["categories"][category] = {
            "eligible_count": len(objects),
            **split,
        }
        manifest["rejected_counts"][category] = len(rejected)
        by_id = {item["object_id"]: item for item in objects}
        for split_name in ("train", "test", "backups"):
            for rank, object_id in enumerate(split[split_name], start=1):
                row = dict(by_id[object_id])
                row["split"] = split_name
                row["rank"] = rank
                csv_rows.append(row)
                manifest["objects"][object_id] = row
        print(
            "{}: eligible={}, train={}, test={}, backups={}".format(
                category, len(objects), len(split["train"]),
                len(split["test"]), len(split["backups"])))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    write_csv(output_csv, csv_rows)
    print("Wrote {}".format(output_json))
    print("Wrote {}".format(output_csv))
    print("SPLIT_RESULT=CANDIDATES_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
