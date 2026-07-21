"""Run the frozen multi-seed L16 robust hyperparameter search."""

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "custom_tools/configs/robust_oa_search.yaml"
OUTPUT = ROOT / "custom_tools/results/evaluations/robust_oa_search"
RUN_ROOT = ROOT / "custom_tools/runs/bc"
CONFIG = ROOT / "custom_tools/configs/unified_student_online_round1.yaml"
INIT = (RUN_ROOT / "unified_student_routed_t70_demo30_seed2025_e40_v1/"
        "epoch=039-step=8640.ckpt")
RESIDUAL = ROOT / "custom_tools/configs/residual_ppo_soup_anchored_gated.yaml"
TUNE_TRAJECTORIES = ROOT / "dexgrasp/dataset/bc_multicategory_valid"
TUNE_SELECTION = ROOT / "custom_tools/configs/bc_train16_heldout_trajectory_validation.yaml"
CONFIRM_TRAJECTORIES = ROOT / "dexgrasp/dataset/unseen_v2_candidates_preprocessed"
CONFIRM_SELECTION = ROOT / "custom_tools/configs/hparam_geometry_confirmation_v1.yaml"
MIB = 1024 ** 2


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--min-free-disk-gib", type=float, default=8.0)
    return parser.parse_args()


def run(command):
    print("RUN: {}".format(" ".join(str(x) for x in command)), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def load_design():
    with DESIGN_PATH.open(encoding="utf-8") as handle:
        design = yaml.safe_load(handle)
    if design.get("status") != "frozen_before_search":
        raise ValueError("Search design is not frozen")
    rows = design["rows"]
    if len(rows) != 16 or any(len(row) != 4 for row in rows):
        raise ValueError("Expected an L16 four-column design")
    # Every pair of columns must contain all 4x4 level pairs exactly once.
    for left in range(4):
        for right in range(left + 1, 4):
            pairs = [(row[left], row[right]) for row in rows]
            if len(set(pairs)) != 16:
                raise ValueError(
                    "Design columns {} and {} are not orthogonal".format(left, right))
    return design


def row_parameters(design, row):
    names = ("teacher_weight", "learning_rate",
             "online_sample_fraction", "noise_value")
    return {name: design["levels"][name][level - 1]
            for name, level in zip(names, row)}


def run_name(row_index, seed):
    return "robust_oa_r{:02d}_seed{}_e20_v1".format(row_index, seed)


def train(row_index, seed, parameters, cli, remaining_trainings):
    name = run_name(row_index, seed)
    run_dir = RUN_ROOT / name
    if (run_dir / "last.ckpt").is_file():
        print("REUSE training: {}".format(name), flush=True)
        return run_dir
    if run_dir.exists() and list(run_dir.glob("*.ckpt")):
        raise RuntimeError("Partial training run exists: {}".format(run_dir))
    free = shutil.disk_usage(ROOT).free
    required = cli.min_free_disk_gib * 1024 ** 3 + remaining_trainings * 210 * MIB
    if free < required:
        raise RuntimeError(
            "Disk guard: {:.1f} GiB free, {:.1f} GiB required for remaining runs"
            .format(free / 1024 ** 3, required / 1024 ** 3))
    run([
        sys.executable, str(ROOT / "custom_tools/train_bc.py"),
        "--config", str(CONFIG), "--run-name", name,
        "--init-checkpoint", str(INIT), "--seed", str(seed),
        "--teacher-weight", str(parameters["teacher_weight"]),
        "--learning-rate", str(parameters["learning_rate"]),
        "--online-sample-fraction", str(parameters["online_sample_fraction"]),
        "--noise-value", str(parameters["noise_value"]),
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
    ])
    return run_dir


def checkpoints(run_dir, epochs):
    result = []
    for epoch in epochs:
        matches = list(run_dir.glob("epoch={:03d}-step=*.ckpt".format(epoch - 1)))
        if len(matches) != 1:
            raise RuntimeError("Expected epoch {} in {}: {}".format(
                epoch, run_dir, matches))
        result.append(matches[0].resolve())
    return result


def metric(item, strict=False):
    prefix = "" if strict else "total_"
    return {
        "success": item["success_count"] if strict else item[prefix + "success_count"],
        "macro": item["macro_success_rate"] if strict else item[
            "macro_official_peak_success_rate"],
        "lift": item["macro_lift_m"] if strict else item["macro_mean_maximum_lift_m"],
        "failure": item["macro_failure_rate"],
    }


def single_rank(item, strict=False):
    value = metric(item, strict)
    return (-value["success"], -value["macro"], value["failure"], -value["lift"])


def robust_rank(values):
    successes = [value["success"] for value in values]
    macros = [value["macro"] for value in values]
    failures = [value["failure"] for value in values]
    return (-min(successes), -statistics.mean(successes),
            -min(macros), -statistics.mean(macros),
            max(failures), statistics.mean(failures))


def coarse_select(row_index, seed, run_dir, epochs, cli):
    directory = OUTPUT / "coarse"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "r{:02d}_seed{}.yaml".format(row_index, seed)
    paths = checkpoints(run_dir, epochs)
    if not output.exists():
        command = [sys.executable,
                   str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py")]
        for path in paths:
            command.extend(["--checkpoint", str(path)])
        command.extend([
            "--allow-stateful-multicheckpoint", "--bc-config", str(CONFIG),
            "--residual-config", str(RESIDUAL),
            "--trajectory-root", str(TUNE_TRAJECTORIES),
            "--object-selection", str(TUNE_SELECTION), "--output", str(output),
            "--seed", "2025", "--min-free-vram-mb", str(cli.min_free_vram_mb),
            "--max-attempts", str(cli.max_attempts),
        ])
        run(command)
    with output.open(encoding="utf-8") as handle:
        rows = yaml.safe_load(handle)["checkpoint_results"]
    for epoch, item in zip(epochs, rows):
        item["requested_epoch"] = epoch
    winner = sorted(rows, key=single_rank)[0]
    return winner, rows, output


def strict_screen(candidates, directory, trajectories, selection, cli):
    command = [sys.executable, str(ROOT / "custom_tools/screen_bc_candidates.py")]
    for label, checkpoint in candidates.items():
        command.extend(["--candidate", "{}={}".format(label, checkpoint)])
    command.extend([
        "--bc-config", str(CONFIG), "--residual-config", str(RESIDUAL),
        "--trajectory-root", str(trajectories),
        "--object-selection", str(selection), "--output-dir", str(directory),
        "--seed", "2025", "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ])
    run(command)
    with (directory / "screen_summary.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["ranking"]


def rank_rows(row_indices, seeds, result_map):
    return sorted(row_indices, key=lambda row: robust_rank([
        metric(result_map[(row, seed)], strict=True) for seed in seeds]))


def main():
    cli = parse_cli()
    design = load_design()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    epochs = design["epochs_screened"]
    parameters = {index: row_parameters(design, row)
                  for index, row in enumerate(design["rows"], 1)}

    stage1_seeds = design["training_seeds_stage1"]
    total_stage1 = len(parameters) * len(stage1_seeds)
    run_dirs = {}
    completed = 0
    for row_index, values in parameters.items():
        for seed in stage1_seeds:
            completed += 1
            run_dirs[(row_index, seed)] = train(
                row_index, seed, values, cli, total_stage1 - completed)

    coarse = {}
    coarse_all = {}
    for key, run_dir in run_dirs.items():
        winner, rows, _ = coarse_select(key[0], key[1], run_dir, epochs, cli)
        coarse[key] = winner
        coarse_all["r{:02d}_seed{}".format(*key)] = rows
    stage1_rank = sorted(parameters, key=lambda row: robust_rank([
        metric(coarse[(row, seed)]) for seed in stage1_seeds]))
    top8 = stage1_rank[:design["selection"]["stage1_strict_top_rows"]]

    strict1_candidates = {
        "r{:02d}_seed{}".format(row, seed): coarse[(row, seed)]["checkpoint"]
        for row in top8 for seed in stage1_seeds}
    strict1 = strict_screen(
        strict1_candidates, OUTPUT / "strict_stage1",
        TUNE_TRAJECTORIES, TUNE_SELECTION, cli)
    strict1_map = {}
    for item in strict1:
        parts = item["label"].split("_seed")
        strict1_map[(int(parts[0][1:]), int(parts[1]))] = item
    strict_stage1_rank = rank_rows(top8, stage1_seeds, strict1_map)
    top4 = strict_stage1_rank[:design["selection"]["stage2_seed2027_top_rows"]]

    seed3 = int(design["training_seed_stage2"])
    coarse3 = {}
    for offset, row in enumerate(top4, 1):
        run_dir = train(row, seed3, parameters[row], cli, len(top4) - offset)
        winner, rows, _ = coarse_select(row, seed3, run_dir, epochs, cli)
        coarse3[row] = winner
        coarse_all["r{:02d}_seed{}".format(row, seed3)] = rows
    strict3_candidates = {
        "r{:02d}_seed{}".format(row, seed3): coarse3[row]["checkpoint"]
        for row in top4}
    strict3 = strict_screen(
        strict3_candidates, OUTPUT / "strict_seed2027",
        TUNE_TRAJECTORIES, TUNE_SELECTION, cli)
    for item in strict3:
        parts = item["label"].split("_seed")
        strict1_map[(int(parts[0][1:]), int(parts[1]))] = item
    all_seeds = stage1_seeds + [seed3]
    stage2_rank = rank_rows(top4, all_seeds, strict1_map)
    top2 = stage2_rank[:design["selection"]["geometry_confirmation_top_rows"]]

    confirm_candidates = {
        "r{:02d}_seed{}".format(row, seed): strict1_map[(row, seed)]["checkpoint"]
        for row in top2 for seed in all_seeds}
    confirmation = strict_screen(
        confirm_candidates, OUTPUT / "geometry_confirmation",
        CONFIRM_TRAJECTORIES, CONFIRM_SELECTION, cli)
    confirm_map = {}
    for item in confirmation:
        parts = item["label"].split("_seed")
        confirm_map[(int(parts[0][1:]), int(parts[1]))] = item
    confirmation_rank = rank_rows(top2, all_seeds, confirm_map)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "design": str(DESIGN_PATH),
        "final_unseen_v2_used": False,
        "parameters_by_row": parameters,
        "stage1_coarse_row_ranking": stage1_rank,
        "stage1_strict_top8": top8,
        "stage1_strict_row_ranking": strict_stage1_rank,
        "stage2_seed2027_rows": top4,
        "stage2_three_seed_row_ranking": stage2_rank,
        "geometry_confirmation_rows": top2,
        "geometry_confirmation_row_ranking": confirmation_rank,
        "selected_robust_row": confirmation_rank[0],
        "selected_robust_parameters": parameters[confirmation_rank[0]],
        "strict_tuning_results": {
            "r{:02d}_seed{}".format(row, seed): item
            for (row, seed), item in strict1_map.items()
        },
        "geometry_confirmation_results": {
            "r{:02d}_seed{}".format(row, seed): item
            for (row, seed), item in confirm_map.items()
        },
        "coarse_results": coarse_all,
        "remaining_disk_gib": shutil.disk_usage(ROOT).free / 1024 ** 3,
    }
    path = OUTPUT / "robust_search_summary.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("ROBUST_OA_SEARCH=COMPLETE summary={}".format(path), flush=True)


if __name__ == "__main__":
    main()
