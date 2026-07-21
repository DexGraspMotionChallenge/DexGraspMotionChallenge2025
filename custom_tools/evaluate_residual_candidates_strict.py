"""Strictly evaluate several residual checkpoints and aggregate their metrics."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", action="append", required=True,
        help="LABEL=RESIDUAL_CHECKPOINT")
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--trajectory-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def absolute(path):
    return Path(path).expanduser().resolve()


def candidates(values):
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError("--candidate must be LABEL=CHECKPOINT")
        label, checkpoint = value.split("=", 1)
        if not label or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                            for char in label):
            raise ValueError("Unsafe label: {}".format(label))
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
        if output.exists():
            print("{} (reuse)".format(label), flush=True)
        else:
            command = [
                sys.executable,
                str(REPO_ROOT / "custom_tools/evaluate_residual_isolated.py"),
                "--residual-checkpoint", str(checkpoint),
                "--bc-checkpoint", str(absolute(cli.bc_checkpoint)),
                "--residual-config", str(absolute(cli.residual_config)),
                "--trajectory-root", str(absolute(cli.trajectory_root)),
                "--trajectory-selection", str(absolute(cli.trajectory_selection)),
                "--num-trajectories", "0",
                "--seed", str(cli.seed),
                "--min-free-vram-mb", str(cli.min_free_vram_mb),
                "--max-attempts", str(cli.max_attempts),
                "--output", str(output),
            ]
            print("strict residual candidate: {}".format(label), flush=True)
            completed = subprocess.run(
                command, cwd=str(REPO_ROOT), check=False)
            if completed.returncode != 0:
                raise RuntimeError("Candidate failed: {}".format(label))
        with output.open(encoding="utf-8") as handle:
            result = yaml.safe_load(handle)
        results.append({
            "label": label,
            "checkpoint": str(checkpoint),
            "total_success_count": result["total_success_count"],
            "total_trajectory_count": result["total_trajectory_count"],
            "overall_official_peak_success_rate": result[
                "overall_official_peak_success_rate"],
            "macro_official_peak_success_rate": result[
                "macro_official_peak_success_rate"],
            "macro_mean_maximum_lift_m": result[
                "macro_mean_maximum_lift_m"],
            "macro_failure_rate": result["macro_failure_rate"],
            "category_macro_success_rates": result[
                "category_macro_success_rates"],
            "output": str(output),
        })
    results.sort(key=lambda item: (
        -item["macro_official_peak_success_rate"],
        -item["macro_mean_maximum_lift_m"],
        item["macro_failure_rate"], item["label"]))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_mode": "fresh_process_per_object_per_candidate",
        "official_success_definition_changed": False,
        "ranked_results": results,
    }
    summary_path = output_dir / "strict_candidates_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("STRICT_RESIDUAL_CANDIDATES=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
