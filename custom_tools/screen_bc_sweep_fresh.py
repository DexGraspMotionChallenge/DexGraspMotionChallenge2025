"""Screen BC epochs with one fresh multi-object process per checkpoint.

This is a ranking-only stage.  Finalists must be reevaluated with one fresh
process per object because the shared multi-object layout changes PhysX
numerics relative to the formal protocol.
"""

import argparse
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", default=[],
        help="LABEL=CHECKPOINT_DIRECTORY (repeat for each noise setting).")
    parser.add_argument(
        "--checkpoint", action="append", default=[],
        help="LABEL=CHECKPOINT for direct candidate screening.")
    parser.add_argument("--epochs", default="20,40,60,80,100")
    parser.add_argument("--bc-config", required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--object-selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def absolute(path):
    return Path(path).expanduser().resolve()


def parse_runs(values):
    runs = []
    for value in values:
        if "=" not in value:
            raise ValueError("--run must be LABEL=DIRECTORY")
        label, directory = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError("Unsafe run label: {}".format(label))
        path = absolute(directory)
        if not path.is_dir():
            raise FileNotFoundError(path)
        runs.append((label, path))
    return runs


def parse_checkpoints(values):
    checkpoints = []
    for value in values:
        if "=" not in value:
            raise ValueError("--checkpoint must be LABEL=CHECKPOINT")
        label, checkpoint = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError("Unsafe checkpoint label: {}".format(label))
        path = absolute(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints.append((label, None, path))
    return checkpoints


def find_checkpoint(directory, epoch):
    matches = sorted(directory.glob("epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one epoch {} checkpoint in {}, got {}".format(
                epoch, directory, matches))
    return matches[0].resolve()


def main():
    cli = parse_cli()
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    epochs = [int(item) for item in cli.epochs.split(",") if item]
    if not epochs or any(epoch < 1 for epoch in epochs):
        raise ValueError("--epochs must contain positive integers")
    runs = parse_runs(cli.run)
    output = absolute(cli.output)
    if output.exists():
        raise FileExistsError(output)
    worker_dir = output.parent / (output.stem + "_fresh_runs")
    worker_dir.mkdir(parents=True, exist_ok=True)
    results = []
    candidates = [
        (label, epoch, find_checkpoint(directory, epoch))
        for label, directory in runs for epoch in epochs]
    candidates.extend(parse_checkpoints(cli.checkpoint))
    if not candidates:
        raise ValueError("Provide --run or --checkpoint")
    for index, (label, epoch, checkpoint) in enumerate(candidates, 1):
        tag = "{}_e{:03d}".format(label, epoch) if epoch is not None else label
        worker_output = worker_dir / (tag + ".yaml")
        if worker_output.exists():
            with worker_output.open(encoding="utf-8") as handle:
                worker = yaml.safe_load(handle)
            print("fresh screen {}/{}: {} (reuse)".format(
                index, len(candidates), tag), flush=True)
        else:
            command = [
                sys.executable,
                str(REPO_ROOT / "custom_tools/evaluate_bc_checkpoints_batched.py"),
                "--checkpoint", str(checkpoint),
                "--bc-config", str(absolute(cli.bc_config)),
                "--residual-config", str(absolute(cli.residual_config)),
                "--trajectory-root", str(absolute(cli.trajectory_root)),
                "--object-selection", str(absolute(cli.object_selection)),
                "--seed", str(cli.seed),
                "--min-free-vram-mb", str(cli.min_free_vram_mb),
                "--output", str(worker_output),
            ]
            print("fresh screen {}/{}: {}".format(
                index, len(candidates), tag), flush=True)
            for attempt in range(1, cli.max_attempts + 1):
                completed = subprocess.run(
                    command, cwd=str(REPO_ROOT), check=False)
                if completed.returncode == 0:
                    break
                print("{} attempt {}/{} failed".format(
                    tag, attempt, cli.max_attempts), flush=True)
            else:
                raise RuntimeError("Screen worker failed: {}".format(tag))
            with worker_output.open(encoding="utf-8") as handle:
                worker = yaml.safe_load(handle)
        item = dict(worker["checkpoint_results"][0])
        item["label"] = label
        item["nominal_epoch"] = epoch
        results.append(item)
    results.sort(key=lambda item: (
        -item["macro_official_peak_success_rate"],
        -item["macro_mean_maximum_lift_m"],
        item["macro_failure_rate"], item["label"],
        item["nominal_epoch"] if item["nominal_epoch"] is not None else -1))
    aggregate = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_mode": "fresh_process_per_checkpoint_shared_multi_object_screen",
        "formal_result": False,
        "warning": "Ranking only; finalists require isolated-object evaluation.",
        "seed": cli.seed,
        "epochs": epochs,
        "ranked_results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(aggregate, handle, allow_unicode=True, sort_keys=False)
    print("FRESH_BC_SWEEP_SCREEN=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
