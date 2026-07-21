"""Finite one-factor search for the unified online student."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "custom_tools/results/evaluations/unified_hparam_search_stage1"
RUN_ROOT = ROOT / "custom_tools/runs/bc"
CONFIG = ROOT / "custom_tools/configs/unified_student_online_round1.yaml"
INIT = (RUN_ROOT / "unified_student_routed_t70_demo30_seed2025_e40_v1/"
        "epoch=039-step=8640.ckpt")
BASELINE = (RUN_ROOT / "unified_student_online_r1_noisefix_seed2025_e20_v1/"
            "epoch=004-step=2140.ckpt")
RESIDUAL = ROOT / "custom_tools/configs/residual_ppo_soup_anchored_gated.yaml"
TRAJECTORIES = ROOT / "dexgrasp/dataset/bc_multicategory_valid"
SELECTION = ROOT / "custom_tools/configs/bc_train16_heldout_trajectory_validation.yaml"
EPOCHS = (5, 10, 15, 20)
SPECS = {
    "lr1e5": ["--learning-rate", "1e-5"],
    "lr4e5": ["--learning-rate", "4e-5"],
    "teacher50": ["--teacher-weight", "0.50"],
    "teacher85": ["--teacher-weight", "0.85"],
    "online33": ["--online-sample-fraction", "0.3333333333"],
    "online67": ["--online-sample-fraction", "0.6666666667"],
}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def run(command):
    print("RUN: {}".format(" ".join(str(x) for x in command)), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def checkpoint(run_dir, epoch):
    matches = list(run_dir.glob("epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(matches) != 1:
        raise RuntimeError("Expected epoch {} in {}: {}".format(epoch, run_dir, matches))
    return matches[0].resolve()


def rank_key(item):
    return (-item["total_success_count"],
            -item["macro_official_peak_success_rate"],
            item["macro_failure_rate"],
            -item["macro_mean_maximum_lift_m"],
            str(item["checkpoint"]))


def main():
    cli = parse_cli()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_dirs = {}
    for label, overrides in SPECS.items():
        run_name = "unified_hparam_{}_seed2025_e20_v1".format(label)
        run_dir = RUN_ROOT / run_name
        run_dirs[label] = run_dir
        if not (run_dir / "last.ckpt").is_file():
            if run_dir.exists() and list(run_dir.glob("*.ckpt")):
                raise RuntimeError("Partial training run exists: {}".format(run_dir))
            run([
                sys.executable, str(ROOT / "custom_tools/train_bc.py"),
                "--config", str(CONFIG), "--run-name", run_name,
                "--init-checkpoint", str(INIT),
                "--min-free-vram-mb", str(cli.min_free_vram_mb),
            ] + overrides)

    coarse_winners = {}
    coarse_summaries = {}
    for label, run_dir in run_dirs.items():
        output = OUTPUT / ("{}_coarse.yaml".format(label))
        checkpoints = [checkpoint(run_dir, epoch) for epoch in EPOCHS]
        if not output.exists():
            command = [
                sys.executable,
                str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py")]
            for path in checkpoints:
                command.extend(["--checkpoint", str(path)])
            command.extend([
                "--allow-stateful-multicheckpoint",
                "--bc-config", str(CONFIG), "--residual-config", str(RESIDUAL),
                "--trajectory-root", str(TRAJECTORIES),
                "--object-selection", str(SELECTION), "--output", str(output),
                "--seed", "2025", "--min-free-vram-mb", str(cli.min_free_vram_mb),
                "--max-attempts", str(cli.max_attempts),
            ])
            run(command)
        with output.open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle)["checkpoint_results"]
        for epoch, item in zip(EPOCHS, values):
            item["requested_epoch"] = epoch
        values.sort(key=rank_key)
        coarse_winners[label] = Path(values[0]["checkpoint"])
        coarse_summaries[label] = values

    strict_dir = OUTPUT / "strict_group_winners"
    command = [sys.executable, str(ROOT / "custom_tools/screen_bc_candidates.py")]
    for label, path in coarse_winners.items():
        command.extend(["--candidate", "{}_best={}".format(label, path)])
    command.extend(["--candidate", "noisefix_baseline={}".format(BASELINE)])
    command.extend([
        "--bc-config", str(CONFIG), "--residual-config", str(RESIDUAL),
        "--trajectory-root", str(TRAJECTORIES),
        "--object-selection", str(SELECTION), "--output-dir", str(strict_dir),
        "--seed", "2025", "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ])
    run(command)
    with (strict_dir / "screen_summary.yaml").open(encoding="utf-8") as handle:
        strict = yaml.safe_load(handle)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "search_type": "predeclared one-factor finite search",
        "final_unseen_v2_used": False,
        "fixed_parameters": {
            "baseline_lr": 2e-5, "baseline_teacher_weight": 0.70,
            "baseline_online_fraction": "approximately 0.5",
            "training_seed": 2025, "epochs_screened": list(EPOCHS),
        },
        "coarse_evaluation_warning": (
            "Within-group epoch selection only; persistent simulator state is not "
            "used for cross-parameter conclusions"),
        "coarse_results": coarse_summaries,
        "coarse_winners": {key: str(value) for key, value in coarse_winners.items()},
        "strict_group_winner_ranking": strict["ranking"],
    }
    path = OUTPUT / "search_summary.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("UNIFIED_HPARAM_SEARCH_STAGE1=COMPLETE summary={}".format(path), flush=True)


if __name__ == "__main__":
    main()
