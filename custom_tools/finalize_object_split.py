"""Freeze a candidate object split after trajectory preprocessing checks."""

import argparse
import copy
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replace low-data training objects with preprocessed backups of the "
            "same geometry group, then freeze the final split."))
    parser.add_argument(
        "--candidate-manifest",
        default=str(REPO_ROOT / "custom_tools" / "configs" /
                    "object_split_candidates.json"))
    parser.add_argument(
        "--preprocess-summary",
        default=str(REPO_ROOT / "custom_tools" / "results" /
                    "object_split_preprocess_summary.json"))
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "custom_tools" / "configs" /
                    "object_split_final.json"))
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "custom_tools" / "configs" /
                    "object_split_final.csv"))
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    candidate_path = Path(args.candidate_manifest).expanduser().resolve()
    summary_path = Path(args.preprocess_summary).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    candidate = load_json(candidate_path)
    summary = load_json(summary_path)
    minimum = int(summary["min_retained_for_bc"])
    checks = {row["object_id"]: row for row in summary["objects"]}
    train_count = int(candidate["criteria"]["train_per_category"])
    test_count = int(candidate["criteria"]["test_per_category"])

    final_categories = {}
    final_objects = {}
    replacements = []
    csv_rows = []

    for category, split in candidate["categories"].items():
        final_train = []
        used = set(split["train"] + split["test"])

        for original_id in split["train"]:
            original_check = checks.get(original_id)
            if original_check is None:
                raise RuntimeError("Missing preprocessing result for {}".format(original_id))

            selected_id = original_id
            if int(original_check["retained_count"]) < minimum:
                proxy = candidate["objects"][original_id]["geometry_proxy"]
                eligible_backups = [
                    object_id for object_id in split["backups"]
                    if object_id not in used
                    and object_id in checks
                    and candidate["objects"][object_id]["geometry_proxy"] == proxy
                    and int(checks[object_id]["retained_count"]) >= minimum
                ]
                if not eligible_backups:
                    raise RuntimeError(
                        "No preprocessed {} backup can replace {}".format(proxy, original_id))
                selected_id = eligible_backups[0]
                used.add(selected_id)
                replacements.append({
                    "category": category,
                    "geometry_proxy": proxy,
                    "replaced_object_id": original_id,
                    "replacement_object_id": selected_id,
                    "replaced_retained_count": int(original_check["retained_count"]),
                    "replacement_retained_count": int(checks[selected_id]["retained_count"]),
                    "reason": "retained trajectories below minimum",
                })
            final_train.append(selected_id)

        final_test = list(split["test"])
        for object_id in final_test:
            check = checks.get(object_id)
            if check is None or int(check["retained_count"]) < minimum:
                raise RuntimeError(
                    "Fixed test object failed preprocessing: {}".format(object_id))

        if len(final_train) != train_count or len(set(final_train)) != train_count:
            raise RuntimeError("Invalid train split for {}".format(category))
        if len(final_test) != test_count or len(set(final_test)) != test_count:
            raise RuntimeError("Invalid test split for {}".format(category))
        if set(final_train) & set(final_test):
            raise RuntimeError("Train/test overlap for {}".format(category))

        final_categories[category] = {"train": final_train, "test": final_test}
        for split_name, object_ids in (("train", final_train), ("test", final_test)):
            for rank, object_id in enumerate(object_ids, start=1):
                metadata = copy.deepcopy(candidate["objects"][object_id])
                check = checks[object_id]
                metadata.update({
                    "split": split_name,
                    "rank": rank,
                    "retained_count": int(check["retained_count"]),
                    "retention_rate": float(check["retention_rate"]),
                    "preprocess_status": check["bc_data_status"],
                })
                final_objects[object_id] = metadata
                csv_rows.append({
                    "category": category,
                    "split": split_name,
                    "rank": rank,
                    "object_id": object_id,
                    "geometry_proxy": metadata["geometry_proxy"],
                    "raw_count": int(check["raw_count"]),
                    "retained_count": int(check["retained_count"]),
                    "retention_rate": float(check["retention_rate"]),
                })

    all_train = [object_id for value in final_categories.values()
                 for object_id in value["train"]]
    all_test = [object_id for value in final_categories.values()
                for object_id in value["test"]]
    if set(all_train) & set(all_test):
        raise RuntimeError("Global train/test overlap")

    final_manifest = {
        "status": "frozen_preflight_passed",
        "source_candidate_manifest": str(candidate_path),
        "source_preprocess_summary": str(summary_path),
        "selection_rule": (
            "Keep the fixed test set. Replace only training objects with fewer "
            "than the minimum retained trajectories, using the first passing "
            "backup from the same geometry group."),
        "min_retained_for_bc": minimum,
        "counts": {
            "train": len(all_train),
            "test": len(all_test),
            "replacements": len(replacements),
        },
        "criteria": candidate["criteria"],
        "categories": final_categories,
        "replacements": replacements,
        "objects": final_objects,
        "metric_note": (
            "retained_count follows the unmodified official task success flag; "
            "relative 30 cm lift remains a separate diagnostic"),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(final_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    for replacement in replacements:
        print(
            "[REPLACED] {category}: {replaced_object_id} ({replaced_retained_count}) "
            "-> {replacement_object_id} ({replacement_retained_count})".format(
                **replacement))
    print("[PASS] final train objects: {}".format(len(all_train)))
    print("[PASS] fixed test objects: {}".format(len(all_test)))
    print("[PASS] all selected objects retain at least {} trajectories".format(minimum))
    print("Wrote {}".format(output_json))
    print("Wrote {}".format(output_csv))
    print("FINAL_SPLIT_RESULT=FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
