"""Evaluate BC checkpoints with one fresh Isaac Gym process per object."""

import argparse
import collections
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--bc-config", required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--object-selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--allow-stateful-multicheckpoint", action="store_true",
        help="Diagnostic only: repeated resets are not selection-grade deterministic.")
    return parser.parse_args()


def absolute(path):
    return Path(path).expanduser().resolve()


def aggregate_checkpoint(checkpoint, object_results):
    objects = [item["objects"][0] for item in object_results]
    total_success = sum(x["official_peak_success_count"] for x in objects)
    total_trajectories = sum(x["trajectory_count"] for x in objects)
    category_rates = collections.defaultdict(list)
    for item in objects:
        category_rates[item["category"]].append(
            item["official_peak_success_rate"])
    return {
        "checkpoint": checkpoint,
        "checkpoint_epoch": object_results[0]["checkpoint_epoch"],
        "total_success_count": total_success,
        "total_trajectory_count": total_trajectories,
        "overall_official_peak_success_rate": total_success / total_trajectories,
        "macro_official_peak_success_rate": sum(
            x["official_peak_success_rate"] for x in objects) / len(objects),
        "macro_mean_maximum_lift_m": sum(
            x["mean_maximum_lift_m"] for x in objects) / len(objects),
        "macro_failure_rate": sum(
            x["failure_rate"] for x in objects) / len(objects),
        "category_macro_success_rates": {
            key: sum(values) / len(values)
            for key, values in sorted(category_rates.items())},
        "objects": objects,
    }


def main():
    cli = parse_cli()
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    output = absolute(cli.output)
    if output.exists():
        raise FileExistsError(output)
    checkpoints = [str(absolute(path)) for path in cli.checkpoint]
    if len(checkpoints) > 1 and not cli.allow_stateful_multicheckpoint:
        raise ValueError(
            "Multiple checkpoints in one persistent simulator are diagnostic only; "
            "use one checkpoint per run or pass --allow-stateful-multicheckpoint.")
    selection_path = absolute(cli.object_selection)
    with selection_path.open(encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)
    object_ids = list(selection["object_ids"])
    per_object_dir = output.parent / (output.stem + "_objects")
    per_object_dir.mkdir(parents=True, exist_ok=True)
    worker_outputs = []
    for index, object_id in enumerate(object_ids, 1):
        object_output = per_object_dir / (object_id + ".yaml")
        if object_output.exists():
            with object_output.open(encoding="utf-8") as handle:
                worker = yaml.safe_load(handle)
            actual = [item["checkpoint"] for item in worker["checkpoint_results"]]
            if actual != checkpoints:
                raise RuntimeError(
                    "Existing worker has a different checkpoint list: {}".format(
                        object_output))
            print("isolated checkpoint evaluation {}/{}: {} (reuse)".format(
                index, len(object_ids), object_id), flush=True)
            worker_outputs.append(worker)
            continue
        command = [
            sys.executable,
            str(REPO_ROOT / "custom_tools/evaluate_bc_checkpoints_batched.py"),
            "--bc-config", str(absolute(cli.bc_config)),
            "--residual-config", str(absolute(cli.residual_config)),
            "--trajectory-root", str(absolute(cli.trajectory_root)),
            "--object-selection", str(selection_path),
            "--object-id", object_id,
            "--seed", str(cli.seed),
            "--min-free-vram-mb", str(cli.min_free_vram_mb),
            "--output", str(object_output),
        ]
        for checkpoint in checkpoints:
            command.extend(["--checkpoint", checkpoint])
        print("isolated checkpoint evaluation {}/{}: {}".format(
            index, len(object_ids), object_id), flush=True)
        for attempt in range(1, cli.max_attempts + 1):
            completed = subprocess.run(
                command, cwd=str(REPO_ROOT), check=False)
            if completed.returncode == 0:
                break
            print("{} attempt {}/{} failed".format(
                object_id, attempt, cli.max_attempts), flush=True)
        else:
            raise RuntimeError("Worker failed: {}".format(object_id))
        with object_output.open(encoding="utf-8") as handle:
            worker_outputs.append(yaml.safe_load(handle))

    checkpoint_results = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        checkpoint_results.append(aggregate_checkpoint(
            checkpoint,
            [worker["checkpoint_results"][checkpoint_index]
             for worker in worker_outputs]))
    aggregate = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_mode": "fresh_process_per_object_multi_checkpoint",
        "success_metric": "official_peak_per_object",
        "official_success_definition_changed": False,
        "seed": cli.seed,
        "trajectory_root": str(absolute(cli.trajectory_root)),
        "object_ids": object_ids,
        "checkpoint_results": checkpoint_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(aggregate, handle, allow_unicode=True, sort_keys=False)
    print("ISOLATED_MULTI_CHECKPOINT_EVALUATION=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
