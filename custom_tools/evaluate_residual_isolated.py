"""Evaluate each object in a fresh process and aggregate official metrics."""

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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--residual-checkpoint")
    mode.add_argument("--zero-residual", action="store_true")
    parser.add_argument("--residual-config", required=True)
    parser.add_argument(
        "--bc-checkpoint", default="",
        help="Optional BC checkpoint override for zero-residual comparisons.")
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--trajectory-selection", required=True)
    parser.add_argument(
        "--use-selection-indices", action="store_true",
        help="Evaluate each object's indices from trajectory-selection.")
    parser.add_argument("--trajectory-start", type=int, default=0)
    parser.add_argument(
        "--num-trajectories", type=int, default=0,
        help="Trajectories per object; 0 evaluates all remaining trajectories.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def resolved(path):
    return Path(path).expanduser().resolve()


def run(cli):
    if cli.max_attempts < 1:
        raise ValueError("--max-attempts must be at least one")
    checkpoint = (
        resolved(cli.residual_checkpoint) if cli.residual_checkpoint else None)
    bc_checkpoint = resolved(cli.bc_checkpoint) if cli.bc_checkpoint else None
    config = resolved(cli.residual_config)
    trajectory_root = resolved(cli.trajectory_root)
    selection_path = resolved(cli.trajectory_selection)
    output = resolved(cli.output)
    if output.exists():
        raise FileExistsError(output)
    with selection_path.open("r", encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)
    object_ids = selection["object_ids"]
    if cli.use_selection_indices and (
            cli.trajectory_start != 0 or cli.num_trajectories != 0):
        raise ValueError(
            "Selection indices cannot be combined with slicing arguments")
    per_object_dir = output.parent / (output.stem + "_objects")
    per_object_dir.mkdir(parents=True, exist_ok=True)

    objects = []
    for index, object_id in enumerate(object_ids, 1):
        object_output = per_object_dir / (object_id + ".yaml")
        if object_output.exists():
            print("isolated evaluation {}/{}: {} (reuse completed)".format(
                index, len(object_ids), object_id), flush=True)
            with object_output.open("r", encoding="utf-8") as handle:
                result = yaml.safe_load(handle)
            if len(result["objects"]) != 1:
                raise RuntimeError("Expected exactly one object result")
            objects.append(result["objects"][0])
            continue
        command = [
            sys.executable,
            str(REPO_ROOT / "custom_tools/evaluate_residual_ppo.py"),
            "--object-id", object_id,
            "--trajectory-root", str(trajectory_root),
            "--trajectory-start", str(cli.trajectory_start),
            "--num-trajectories", str(cli.num_trajectories),
            "--residual-config", str(config),
            "--seed", str(cli.seed),
            "--min-free-vram-mb", str(cli.min_free_vram_mb),
            "--output", str(object_output),
        ]
        if bc_checkpoint is not None:
            command.extend(["--bc-checkpoint", str(bc_checkpoint)])
        if cli.use_selection_indices:
            selected = selection["trajectory_indices_by_object"][object_id]
            command.extend([
                "--trajectory-indices", ",".join(str(i) for i in selected)])
        if cli.zero_residual:
            command.append("--zero-residual")
        else:
            command.extend(["--residual-checkpoint", str(checkpoint)])
        print("isolated evaluation {}/{}: {}".format(
            index, len(object_ids), object_id), flush=True)
        for attempt in range(1, cli.max_attempts + 1):
            completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
            if completed.returncode == 0:
                break
            print("object {} attempt {}/{} failed".format(
                object_id, attempt, cli.max_attempts), flush=True)
        else:
            raise RuntimeError(
                "Object evaluation failed after {} attempts: {}".format(
                    cli.max_attempts, object_id))
        with object_output.open("r", encoding="utf-8") as handle:
            result = yaml.safe_load(handle)
        if len(result["objects"]) != 1:
            raise RuntimeError("Expected exactly one object result")
        objects.append(result["objects"][0])

    category_rates = collections.defaultdict(list)
    for item in objects:
        category = item["object_id"].split("-", 2)[1]
        category_rates[category].append(item["official_peak_success_rate"])
    total_successes = sum(
        item["official_peak_success_count"] for item in objects)
    total_trajectories = sum(item["trajectory_count"] for item in objects)
    aggregate = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_mode": "fresh_process_per_object",
        "success_metric": "official_peak",
        "official_success_definition_changed": False,
        "evaluation_policy": (
            "zero_residual_bc_control" if cli.zero_residual
            else "deterministic_residual_policy"),
        "residual_checkpoint": str(checkpoint) if checkpoint else None,
        "bc_checkpoint": str(bc_checkpoint) if bc_checkpoint else None,
        "residual_config": str(config),
        "trajectory_root": str(trajectory_root),
        "trajectory_selection": str(selection_path),
        "uses_selection_indices": cli.use_selection_indices,
        "trajectory_start": cli.trajectory_start,
        "num_trajectories": cli.num_trajectories,
        "seed": cli.seed,
        "total_success_count": total_successes,
        "total_trajectory_count": total_trajectories,
        "overall_official_peak_success_rate": (
            total_successes / total_trajectories),
        "macro_official_peak_success_rate": sum(
            item["official_peak_success_rate"] for item in objects) / len(objects),
        "macro_mean_maximum_lift_m": sum(
            item["diagnostic_mean_maximum_lift_m"] for item in objects)
            / len(objects),
        "macro_failure_rate": sum(
            item["diagnostic_failure_rate"] for item in objects) / len(objects),
        "category_macro_success_rates": {
            category: sum(values) / len(values)
            for category, values in sorted(category_rates.items())
        },
        "objects": objects,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(aggregate, handle, allow_unicode=True, sort_keys=False)
    print("Saved isolated aggregate: {}".format(output), flush=True)
    return output


if __name__ == "__main__":
    run(parse_cli())
