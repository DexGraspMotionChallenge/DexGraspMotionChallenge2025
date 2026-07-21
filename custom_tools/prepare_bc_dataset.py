"""Create leakage-free BC train/validation data from the frozen object split."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_KEYS = {
    "obs",
    "vis_unscale_actions",
    "unscale_actions",
    "obj_rotmat",
    "obj_scale",
    "grasp_seqs",
    "hand_pcds",
    "obj_pcds",
    "h2o_vec",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split each selected training object by whole trajectories and keep "
            "the fixed unseen objects in a separate directory."))
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "custom_tools" / "configs" /
                    "object_split_final.json"))
    parser.add_argument(
        "--source-root",
        default=str(REPO_ROOT / "dexgrasp" / "dataset" /
                    "object_split_candidates_preprocessed"))
    parser.add_argument(
        "--train-root",
        default=str(REPO_ROOT / "dexgrasp" / "dataset" /
                    "bc_multicategory_train"))
    parser.add_argument(
        "--valid-root",
        default=str(REPO_ROOT / "dexgrasp" / "dataset" /
                    "bc_multicategory_valid"))
    parser.add_argument(
        "--unseen-root",
        default=str(REPO_ROOT / "dexgrasp" / "dataset" /
                    "object_split_final_unseen"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--output-summary",
        default=str(REPO_ROOT / "custom_tools" / "results" /
                    "bc_dataset_split_summary.json"))
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "custom_tools" / "results" /
                    "bc_dataset_split_summary.csv"))
    return parser.parse_args()


def stable_rng(seed, object_id):
    digest = hashlib.sha256("{}:{}".format(seed, object_id).encode("utf-8")).digest()
    object_seed = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return np.random.RandomState(object_seed)


def check_empty_output(root):
    root.mkdir(parents=True, exist_ok=True)
    existing = list(root.iterdir())
    if existing:
        raise FileExistsError(
            "Output directory is not empty: {}. Use a new directory so an old "
            "split cannot be mixed into this experiment.".format(root))


def subset_data(data, local_indices, split_name, seed):
    local_indices = np.asarray(local_indices, dtype=np.int64)
    sequence_count = int(len(data["grasp_seqs"]))
    raw_indices = np.asarray(
        data.get("official_final_success_idx", data.get("success_idx")),
        dtype=np.int64)
    if len(raw_indices) != sequence_count:
        raise ValueError("Official success indices do not align with retained sequences")
    selected_raw_indices = raw_indices[local_indices]

    output = {}
    for key, value in data.items():
        if key in SEQUENCE_KEYS:
            if not isinstance(value, np.ndarray) or len(value) != sequence_count:
                raise ValueError("Sequence field has invalid length: {}".format(key))
            output[key] = value[local_indices]
        else:
            output[key] = value

    output["success_idx"] = selected_raw_indices.copy()
    output["official_final_success_idx"] = selected_raw_indices.copy()
    for key in ("ever_task_success_idx", "lift_30cm_idx"):
        if key in data:
            output[key] = np.intersect1d(
                np.asarray(data[key], dtype=np.int64), selected_raw_indices)
    if "maximum_lift" in data:
        maximum_lift = np.asarray(data["maximum_lift"])
        output["maximum_lift"] = maximum_lift[selected_raw_indices]
    output["custom_split_info"] = {
        "split": split_name,
        "seed": int(seed),
        "source_retained_count": sequence_count,
        "selected_count": int(len(local_indices)),
        "selected_local_indices": local_indices.tolist(),
        "selected_raw_indices": selected_raw_indices.tolist(),
    }
    return output


def main():
    args = parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")

    manifest_path = Path(args.manifest).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    train_root = Path(args.train_root).expanduser().resolve()
    valid_root = Path(args.valid_root).expanduser().resolve()
    unseen_root = Path(args.unseen_root).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "frozen_preflight_passed":
        raise RuntimeError("Refusing to prepare data from a non-frozen manifest")

    for root in (train_root, valid_root, unseen_root):
        check_empty_output(root)

    rows = []
    for category, category_split in manifest["categories"].items():
        for object_id in category_split["train"]:
            source = source_root / (object_id + ".npy")
            if not source.is_file():
                raise FileNotFoundError(source)
            data = np.load(str(source), allow_pickle=True).item()
            count = int(len(data["grasp_seqs"]))
            valid_count = max(2, int(round(count * args.validation_fraction)))
            if count - valid_count < 2:
                raise RuntimeError("Too few training trajectories for {}".format(object_id))

            permutation = stable_rng(args.seed, object_id).permutation(count)
            valid_indices = np.sort(permutation[:valid_count])
            train_indices = np.sort(permutation[valid_count:])
            train_data = subset_data(data, train_indices, "train", args.seed)
            valid_data = subset_data(data, valid_indices, "valid", args.seed)
            np.save(str(train_root / (object_id + ".npy")), train_data)
            np.save(str(valid_root / (object_id + ".npy")), valid_data)
            rows.append({
                "category": category,
                "object_id": object_id,
                "source_count": count,
                "train_count": int(len(train_indices)),
                "valid_count": int(len(valid_indices)),
            })

        for object_id in category_split["test"]:
            source = source_root / (object_id + ".npy")
            destination = unseen_root / (object_id + ".npy")
            if not source.is_file():
                raise FileNotFoundError(source)
            os.symlink(str(source), str(destination))

    train_ids = {path.stem for path in train_root.glob("*.npy")}
    valid_ids = {path.stem for path in valid_root.glob("*.npy")}
    unseen_ids = {path.stem for path in unseen_root.glob("*.npy")}
    if train_ids != valid_ids:
        raise RuntimeError("Train and validation object IDs do not match")
    if (train_ids | valid_ids) & unseen_ids:
        raise RuntimeError("Unseen objects leaked into BC train/validation data")

    summary = {
        "status": "ready",
        "manifest": str(manifest_path),
        "source_root": str(source_root),
        "train_root": str(train_root),
        "valid_root": str(valid_root),
        "unseen_root": str(unseen_root),
        "seed": int(args.seed),
        "validation_fraction": float(args.validation_fraction),
        "train_object_count": len(train_ids),
        "unseen_object_count": len(unseen_ids),
        "train_trajectory_count": sum(row["train_count"] for row in rows),
        "valid_trajectory_count": sum(row["valid_count"] for row in rows),
        "objects": rows,
        "leakage_check": "PASS",
    }
    output_summary = Path(args.output_summary).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("[PASS] BC train objects: {}; trajectories: {}".format(
        len(train_ids), summary["train_trajectory_count"]))
    print("[PASS] BC validation objects: {}; trajectories: {}".format(
        len(valid_ids), summary["valid_trajectory_count"]))
    print("[PASS] fixed unseen objects: {}".format(len(unseen_ids)))
    print("[PASS] unseen-object leakage check")
    print("Wrote {}".format(output_summary))
    print("Wrote {}".format(output_csv))
    print("BC_DATASET_RESULT=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
