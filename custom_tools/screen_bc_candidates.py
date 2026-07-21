"""Strictly screen labeled BC candidates with fresh processes per object."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--bc-config", required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--object-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def absolute(value):
    return Path(value).expanduser().resolve()


def candidates(values):
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError("--candidate must be LABEL=CHECKPOINT")
        label, checkpoint = value.split("=", 1)
        if not label or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                            for char in label):
            raise ValueError("Unsafe candidate label: {}".format(label))
        checkpoint = absolute(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        parsed.append((label, checkpoint))
    return parsed


def main():
    cli = parse_cli()
    output_dir = absolute(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for label, checkpoint in candidates(cli.candidate):
        output = output_dir / (label + ".yaml")
        if not output.exists():
            command = [
                sys.executable,
                str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
                "--checkpoint", str(checkpoint),
                "--bc-config", str(absolute(cli.bc_config)),
                "--residual-config", str(absolute(cli.residual_config)),
                "--trajectory-root", str(absolute(cli.trajectory_root)),
                "--object-selection", str(absolute(cli.object_selection)),
                "--output", str(output), "--seed", str(cli.seed),
                "--min-free-vram-mb", str(cli.min_free_vram_mb),
                "--max-attempts", str(cli.max_attempts),
            ]
            subprocess.run(command, cwd=str(ROOT), check=True)
        with output.open(encoding="utf-8") as handle:
            item = yaml.safe_load(handle)["checkpoint_results"][0]
        results.append({
            "label": label, "checkpoint": str(checkpoint), "output": str(output),
            "success_count": item["total_success_count"],
            "trajectory_count": item["total_trajectory_count"],
            "macro_success_rate": item["macro_official_peak_success_rate"],
            "macro_lift_m": item["macro_mean_maximum_lift_m"],
            "macro_failure_rate": item["macro_failure_rate"],
            "category_macro_success_rates": item.get("category_macro_success_rates"),
        })
    results.sort(key=lambda item: (
        -item["success_count"], -item["macro_success_rate"],
        item["macro_failure_rate"], -item["macro_lift_m"], item["label"]))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_mode": "fresh_process_per_object_per_checkpoint",
        "selection_rule": "success count, macro success, failure rate, lift",
        "final_unseen_v2_used": False,
        "ranking": results,
    }
    path = output_dir / "screen_summary.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("BC_CANDIDATE_SCREEN=COMPLETE summary={}".format(path), flush=True)


if __name__ == "__main__":
    main()
