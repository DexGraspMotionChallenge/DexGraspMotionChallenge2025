"""Summarize trajectory retention for a proposed object split."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize official preprocessing results for split candidates.")
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "custom_tools" / "configs" /
                    "object_split_candidates.json"))
    parser.add_argument("--preprocessed-root", required=True)
    parser.add_argument("--min-retained", type=int, default=12)
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "custom_tools" / "results" /
                    "object_split_preprocess_summary.json"))
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "custom_tools" / "results" /
                    "object_split_preprocess_summary.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    preprocessed_root = Path(args.preprocessed_root).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    rows = []
    for path in sorted(preprocessed_root.glob("*.npy")):
        if path.stem not in manifest["objects"]:
            continue
        metadata = manifest["objects"][path.stem]
        data = np.load(str(path), allow_pickle=True).item()
        raw_count = int(len(data["maximum_lift"]))
        retained_count = int(len(data["grasp_seqs"]))
        ever_count = int(len(data["ever_task_success_idx"]))
        relative_lift_30cm_count = int(len(data["lift_30cm_idx"]))
        row = {
            "category": metadata["category"],
            "split": metadata["split"],
            "geometry_proxy": metadata["geometry_proxy"],
            "object_id": path.stem,
            "raw_count": raw_count,
            "retained_count": retained_count,
            "retention_rate": retained_count / raw_count if raw_count else 0.0,
            "ever_task_success_count": ever_count,
            "relative_lift_30cm_count": relative_lift_30cm_count,
            "mean_maximum_relative_lift": float(np.mean(data["maximum_lift"])),
            "bc_data_status": "PASS" if retained_count >= args.min_retained else "REPLACE",
        }
        rows.append(row)

    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "manifest": str(manifest_path),
        "preprocessed_root": str(preprocessed_root),
        "min_retained_for_bc": args.min_retained,
        "pass_count": sum(row["bc_data_status"] == "PASS" for row in rows),
        "replace_count": sum(row["bc_data_status"] == "REPLACE" for row in rows),
        "objects": rows,
        "metric_note": (
            "retained_count follows the unmodified official task success flag; "
            "relative_lift_30cm_count is diagnostic only"),
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    fieldnames = list(rows[0]) if rows else []
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            "[{status}] {category}/{split} {object_id}: retained={retained_count}/{raw_count}"
            .format(status=row["bc_data_status"], **row))
    print("Wrote {}".format(output_json))
    print("Wrote {}".format(output_csv))
    print("PREPROCESS_SUMMARY_RESULT=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
