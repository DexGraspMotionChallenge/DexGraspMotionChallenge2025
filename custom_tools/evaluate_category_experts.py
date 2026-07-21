"""Strictly screen category-specialized BC checkpoints and combine experts."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("bottle", "mug", "bowl", "camera")


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expert", action="append", required=True,
        help="CATEGORY=RUN_DIRECTORY; provide all four categories")
    parser.add_argument("--epochs", default="10,20,40")
    parser.add_argument("--manifest", required=True)
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


def parse_experts(values):
    experts = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--expert must be CATEGORY=RUN_DIRECTORY")
        category, run_dir = value.split("=", 1)
        if category not in CATEGORIES or category in experts:
            raise ValueError("Invalid or duplicate category: {}".format(category))
        run_dir = absolute(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        experts[category] = run_dir
    if set(experts) != set(CATEGORIES):
        raise ValueError("Exactly four category experts are required")
    return experts


def checkpoint_for_epoch(run_dir, epoch):
    matches = list(run_dir.glob("epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one epoch {} checkpoint in {}, got {}".format(
                epoch, run_dir, len(matches)))
    return matches[0].resolve()


def main():
    cli = parse_cli()
    experts = parse_experts(cli.expert)
    epochs = [int(value) for value in cli.epochs.split(",")]
    if not epochs or any(epoch <= 0 for epoch in epochs):
        raise ValueError("--epochs must contain positive integers")
    with absolute(cli.manifest).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    output_dir = absolute(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked_by_category = {}
    for category in CATEGORIES:
        selection = {
            "status": "frozen_category_expert_validation_selection",
            "category": category,
            "object_ids": manifest["categories"][category]["train"],
        }
        selection_path = output_dir / (category + "_selection.yaml")
        with selection_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(selection, handle, sort_keys=False)
        candidates = []
        for epoch in epochs:
            checkpoint = checkpoint_for_epoch(experts[category], epoch)
            output = output_dir / ("{}_epoch{:03d}.yaml".format(category, epoch))
            if not output.exists():
                command = [
                    sys.executable,
                    str(REPO_ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
                    "--checkpoint", str(checkpoint),
                    "--bc-config", str(absolute(cli.bc_config)),
                    "--residual-config", str(absolute(cli.residual_config)),
                    "--trajectory-root", str(absolute(cli.trajectory_root)),
                    "--object-selection", str(selection_path),
                    "--seed", str(cli.seed),
                    "--min-free-vram-mb", str(cli.min_free_vram_mb),
                    "--max-attempts", str(cli.max_attempts),
                    "--output", str(output),
                ]
                print("strict category candidate: {} epoch {}".format(
                    category, epoch), flush=True)
                completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
                if completed.returncode != 0:
                    raise RuntimeError(
                        "Category evaluation failed: {} epoch {}".format(
                            category, epoch))
            with output.open(encoding="utf-8") as handle:
                result = yaml.safe_load(handle)["checkpoint_results"][0]
            result["requested_epoch"] = epoch
            result["output"] = str(output)
            candidates.append(result)
        candidates.sort(key=lambda item: (
            -item["macro_official_peak_success_rate"],
            -item["macro_mean_maximum_lift_m"],
            item["macro_failure_rate"], item["requested_epoch"]))
        ranked_by_category[category] = candidates

    selected = {category: ranked_by_category[category][0]
                for category in CATEGORIES}
    selected_objects = [item for category in CATEGORIES
                        for item in selected[category]["objects"]]
    combined = {
        "total_success_count": sum(
            item["official_peak_success_count"] for item in selected_objects),
        "total_trajectory_count": sum(
            item["trajectory_count"] for item in selected_objects),
        "macro_official_peak_success_rate": sum(
            item["official_peak_success_rate"] for item in selected_objects)
            / len(selected_objects),
        "macro_mean_maximum_lift_m": sum(
            item["mean_maximum_lift_m"] for item in selected_objects)
            / len(selected_objects),
        "macro_failure_rate": sum(
            item["failure_rate"] for item in selected_objects)
            / len(selected_objects),
        "selected_experts": {
            category: {
                "requested_epoch": item["requested_epoch"],
                "checkpoint": item["checkpoint"],
                "success_count": item["total_success_count"],
                "trajectory_count": item["total_trajectory_count"],
                "macro_success_rate": item["macro_official_peak_success_rate"],
            } for category, item in selected.items()
        },
    }
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_mode": "fresh_process_per_object_per_category_checkpoint",
        "official_success_definition_changed": False,
        "selection_epochs_predeclared": epochs,
        "ranked_by_category": ranked_by_category,
        "combined_selected_experts": combined,
    }
    with (output_dir / "category_experts_summary.yaml").open(
            "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("CATEGORY_EXPERT_EVALUATION=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
