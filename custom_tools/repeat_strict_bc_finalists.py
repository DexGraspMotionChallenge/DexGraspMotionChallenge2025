"""Run repeated formal BC evaluations for a small finalist set."""

import argparse
from datetime import datetime
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", action="append", required=True,
        help="LABEL=CHECKPOINT (repeat for each finalist/control).")
    parser.add_argument("--repeat-start", type=int, default=2)
    parser.add_argument("--repeat-end", type=int, default=3)
    parser.add_argument("--bc-config", required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--object-selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def absolute(path):
    return Path(path).expanduser().resolve()


def parse_candidates(values):
    candidates = []
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
        candidates.append((label, checkpoint))
    return candidates


def main():
    cli = parse_cli()
    if cli.repeat_start < 1 or cli.repeat_end < cli.repeat_start:
        raise ValueError("Invalid repeat range")
    candidates = parse_candidates(cli.candidate)
    output_dir = absolute(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {label: [] for label, _ in candidates}
    for label, checkpoint in candidates:
        for repeat in range(cli.repeat_start, cli.repeat_end + 1):
            output = output_dir / ("{}_r{}.yaml".format(label, repeat))
            if output.exists():
                print("{} repeat {} (reuse)".format(label, repeat), flush=True)
            else:
                command = [
                    sys.executable,
                    str(REPO_ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
                    "--checkpoint", str(checkpoint),
                    "--bc-config", str(absolute(cli.bc_config)),
                    "--residual-config", str(absolute(cli.residual_config)),
                    "--trajectory-root", str(absolute(cli.trajectory_root)),
                    "--object-selection", str(absolute(cli.object_selection)),
                    "--seed", str(cli.seed),
                    "--min-free-vram-mb", str(cli.min_free_vram_mb),
                    "--max-attempts", str(cli.max_attempts),
                    "--output", str(output),
                ]
                print("{} repeat {}".format(label, repeat), flush=True)
                completed = subprocess.run(
                    command, cwd=str(REPO_ROOT), check=False)
                if completed.returncode != 0:
                    raise RuntimeError(
                        "Strict evaluation failed: {} repeat {}".format(
                            label, repeat))
            with output.open(encoding="utf-8") as handle:
                result = yaml.safe_load(handle)["checkpoint_results"][0]
            all_results[label].append({
                "repeat": repeat,
                "output": str(output),
                "total_success_count": result["total_success_count"],
                "macro_official_peak_success_rate": result[
                    "macro_official_peak_success_rate"],
                "macro_mean_maximum_lift_m": result[
                    "macro_mean_maximum_lift_m"],
                "macro_failure_rate": result["macro_failure_rate"],
            })
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_mode": "repeated_fresh_process_per_object",
        "repeat_range": [cli.repeat_start, cli.repeat_end],
        "candidates": {},
    }
    for label, values in all_results.items():
        successes = [item["total_success_count"] for item in values]
        macros = [item["macro_official_peak_success_rate"] for item in values]
        summary["candidates"][label] = {
            "runs": values,
            "mean_success_count": statistics.mean(successes),
            "success_count_population_std": statistics.pstdev(successes),
            "mean_macro_success_rate": statistics.mean(macros),
        }
    summary_path = output_dir / "repeats_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("STRICT_BC_FINALIST_REPEATS=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
