"""Repeated strict evaluation of a fixed category-routed BC teacher pool."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("bottle", "mug", "bowl", "camera")


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher", action="append", required=True,
        help="CATEGORY=CHECKPOINT; provide all four categories")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-split", choices=("train", "test"),
                        default="train")
    parser.add_argument("--bc-config", required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def absolute(path):
    return Path(path).expanduser().resolve()


def teachers(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--teacher must be CATEGORY=CHECKPOINT")
        category, checkpoint = value.split("=", 1)
        if category not in CATEGORIES or category in result:
            raise ValueError("Invalid or duplicate category: {}".format(category))
        checkpoint = absolute(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result[category] = checkpoint
    if set(result) != set(CATEGORIES):
        raise ValueError("Exactly four category teachers are required")
    return result


def main():
    cli = parse_cli()
    if cli.repeats < 1:
        raise ValueError("--repeats must be positive")
    teacher_map = teachers(cli.teacher)
    with absolute(cli.manifest).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    output_dir = absolute(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = {}
    for category in CATEGORIES:
        selection_path = output_dir / (category + "_selection.yaml")
        selection = {
            "status": "frozen_routed_teacher_validation_selection",
            "category": category,
            "manifest_split": cli.manifest_split,
            "object_ids": manifest["categories"][category][cli.manifest_split],
        }
        if not selection["object_ids"]:
            raise ValueError(
                "No {} objects for {}".format(cli.manifest_split, category))
        with selection_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(selection, handle, sort_keys=False)
        selections[category] = selection_path

    repeat_results = []
    for repeat in range(1, cli.repeats + 1):
        category_results = {}
        all_objects = []
        for category in CATEGORIES:
            output = output_dir / ("repeat{}_{}.yaml".format(repeat, category))
            if not output.exists():
                command = [
                    sys.executable,
                    str(REPO_ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
                    "--checkpoint", str(teacher_map[category]),
                    "--bc-config", str(absolute(cli.bc_config)),
                    "--residual-config", str(absolute(cli.residual_config)),
                    "--trajectory-root", str(absolute(cli.trajectory_root)),
                    "--object-selection", str(selections[category]),
                    "--seed", str(cli.seed),
                    "--min-free-vram-mb", str(cli.min_free_vram_mb),
                    "--max-attempts", str(cli.max_attempts),
                    "--output", str(output),
                ]
                print("routed repeat {}: {}".format(repeat, category), flush=True)
                completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
                if completed.returncode != 0:
                    raise RuntimeError(
                        "Routed evaluation failed: repeat {} {}".format(
                            repeat, category))
            with output.open(encoding="utf-8") as handle:
                item = yaml.safe_load(handle)["checkpoint_results"][0]
            category_results[category] = {
                "checkpoint": item["checkpoint"],
                "success_count": item["total_success_count"],
                "trajectory_count": item["total_trajectory_count"],
                "macro_success_rate": item["macro_official_peak_success_rate"],
                "macro_lift_m": item["macro_mean_maximum_lift_m"],
                "macro_failure_rate": item["macro_failure_rate"],
                "output": str(output),
            }
            all_objects.extend(item["objects"])
        repeat_results.append({
            "repeat": repeat,
            "total_success_count": sum(
                item["official_peak_success_count"] for item in all_objects),
            "total_trajectory_count": sum(
                item["trajectory_count"] for item in all_objects),
            "macro_official_peak_success_rate": sum(
                item["official_peak_success_rate"] for item in all_objects)
                / len(all_objects),
            "macro_mean_maximum_lift_m": sum(
                item["mean_maximum_lift_m"] for item in all_objects)
                / len(all_objects),
            "macro_failure_rate": sum(
                item["failure_rate"] for item in all_objects)
                / len(all_objects),
            "categories": category_results,
        })

    success_counts = [item["total_success_count"] for item in repeat_results]
    macro_rates = [item["macro_official_peak_success_rate"]
                   for item in repeat_results]
    summary = {
        "evaluation_mode": "fixed_router_fresh_process_per_object",
        "official_success_definition_changed": False,
        "teacher_checkpoints": {
            category: str(path) for category, path in teacher_map.items()},
        "repeats": repeat_results,
        "aggregate": {
            "mean_success_count": statistics.mean(success_counts),
            "success_count_population_std": statistics.pstdev(success_counts),
            "mean_macro_success_rate": statistics.mean(macro_rates),
            "macro_success_rate_population_std": statistics.pstdev(macro_rates),
            "mean_macro_lift_m": statistics.mean(
                item["macro_mean_maximum_lift_m"] for item in repeat_results),
            "mean_macro_failure_rate": statistics.mean(
                item["macro_failure_rate"] for item in repeat_results),
        },
    }
    with (output_dir / "routed_repeats_summary.yaml").open(
            "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("ROUTED_BC_REPEATS=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
