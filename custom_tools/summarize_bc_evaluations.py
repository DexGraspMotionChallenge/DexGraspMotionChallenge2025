"""Create report-ready BC checkpoint comparison tables and a grouped bar plot."""

import argparse
import collections
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.result_file) != len(args.label):
        raise ValueError("Provide one --label for every --result-file")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    categories = ["bottle", "mug", "bowl", "camera"]
    summary_rows = []
    object_rows = []
    for label, result_path in zip(args.label, args.result_file):
        with result_path.open("r", encoding="utf-8") as handle:
            result = yaml.safe_load(handle)
        trial_count = len(result.get("trajectory_indices", []))
        if trial_count == 0:
            raise ValueError(
                "Result does not record fixed trajectory indices: {}".format(result_path))
        category_rates = collections.defaultdict(list)
        for item in result["objects"]:
            category = item["object_id"].split("-", 2)[1]
            category_rates[category].append(float(item["success_rate"]))
            object_rows.append({
                "label": label,
                "object_id": item["object_id"],
                "category": category,
                "success_count": int(item["success_count"]),
                "trial_count": trial_count,
                "success_rate": float(item["success_rate"]),
            })
        row = {
            "label": label,
            "checkpoint_epoch": result.get("checkpoint_epoch"),
            "checkpoint_global_step": result.get("checkpoint_global_step"),
            "overall_success_rate": float(result["total_succ_rates"]),
        }
        for category in categories:
            values = category_rates[category]
            row[category + "_success_rate"] = sum(values) / len(values)
        summary_rows.append(row)

    summary_path = args.output_dir / "bc_checkpoint_comparison.csv"
    object_path = args.output_dir / "bc_object_results.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with object_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(object_rows[0]))
        writer.writeheader()
        writer.writerows(object_rows)

    groups = ["overall"] + categories
    x_positions = np.arange(len(groups))
    width = 0.8 / len(summary_rows)
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    for index, row in enumerate(summary_rows):
        rates = [row["overall_success_rate"]] + [
            row[category + "_success_rate"] for category in categories]
        offset = (index - (len(summary_rows) - 1) / 2.0) * width
        axis.bar(x_positions + offset, np.asarray(rates) * 100.0,
                 width=width, label=row["label"])
    axis.set_xticks(x_positions)
    axis.set_xticklabels(groups)
    axis.set_ylabel("Official success rate (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    plot_path = args.output_dir / "bc_checkpoint_comparison.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    print(summary_path.resolve())
    print(object_path.resolve())
    print(plot_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
