"""Check whether the 85% teacher-weight gain survives a new fine-tuning seed."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "unified_hparam_teacher85_seed2026_e20_v1"
RUN_DIR = ROOT / "custom_tools/runs/bc" / RUN_NAME
OUTPUT = ROOT / "custom_tools/results/evaluations/teacher85_seed2026_check"
CONFIG = ROOT / "custom_tools/configs/unified_student_online_round1.yaml"
INIT = (ROOT / "custom_tools/runs/bc/unified_student_routed_t70_demo30_seed2025_e40_v1/"
        "epoch=039-step=8640.ckpt")
SEED2025 = (ROOT / "custom_tools/runs/bc/unified_hparam_teacher85_seed2025_e20_v1/"
            "epoch=004-step=2140.ckpt")
BASELINE = (ROOT / "custom_tools/runs/bc/unified_student_online_r1_noisefix_seed2025_e20_v1/"
            "epoch=004-step=2140.ckpt")
RESIDUAL = ROOT / "custom_tools/configs/residual_ppo_soup_anchored_gated.yaml"
TRAJECTORIES = ROOT / "dexgrasp/dataset/bc_multicategory_valid"
SELECTION = ROOT / "custom_tools/configs/bc_train16_heldout_trajectory_validation.yaml"
EPOCHS = (5, 10, 15, 20)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def run(command):
    print("RUN: {}".format(" ".join(str(x) for x in command)), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def checkpoint(epoch):
    matches = list(RUN_DIR.glob("epoch={:03d}-step=*.ckpt".format(epoch - 1)))
    if len(matches) != 1:
        raise RuntimeError("Expected epoch {} checkpoint: {}".format(epoch, matches))
    return matches[0].resolve()


def rank_key(item):
    return (-item["total_success_count"],
            -item["macro_official_peak_success_rate"],
            item["macro_failure_rate"],
            -item["macro_mean_maximum_lift_m"])


def main():
    cli = parse_cli()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not (RUN_DIR / "last.ckpt").is_file():
        if RUN_DIR.exists() and list(RUN_DIR.glob("*.ckpt")):
            raise RuntimeError("Partial run exists: {}".format(RUN_DIR))
        run([
            sys.executable, str(ROOT / "custom_tools/train_bc.py"),
            "--config", str(CONFIG), "--run-name", RUN_NAME,
            "--init-checkpoint", str(INIT), "--seed", "2026",
            "--teacher-weight", "0.85",
            "--min-free-vram-mb", str(cli.min_free_vram_mb),
        ])
    coarse = OUTPUT / "seed2026_coarse.yaml"
    paths = [checkpoint(epoch) for epoch in EPOCHS]
    if not coarse.exists():
        command = [
            sys.executable,
            str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py")]
        for path in paths:
            command.extend(["--checkpoint", str(path)])
        command.extend([
            "--allow-stateful-multicheckpoint", "--bc-config", str(CONFIG),
            "--residual-config", str(RESIDUAL),
            "--trajectory-root", str(TRAJECTORIES),
            "--object-selection", str(SELECTION), "--output", str(coarse),
            "--seed", "2025", "--min-free-vram-mb", str(cli.min_free_vram_mb),
            "--max-attempts", str(cli.max_attempts),
        ])
        run(command)
    with coarse.open(encoding="utf-8") as handle:
        coarse_rows = yaml.safe_load(handle)["checkpoint_results"]
    for epoch, item in zip(EPOCHS, coarse_rows):
        item["requested_epoch"] = epoch
    winner = sorted(coarse_rows, key=rank_key)[0]
    strict_dir = OUTPUT / "strict_comparison"
    run([
        sys.executable, str(ROOT / "custom_tools/screen_bc_candidates.py"),
        "--candidate", "teacher85_seed2026_best={}".format(winner["checkpoint"]),
        "--candidate", "teacher85_seed2025_e05={}".format(SEED2025),
        "--candidate", "teacher70_baseline={}".format(BASELINE),
        "--bc-config", str(CONFIG), "--residual-config", str(RESIDUAL),
        "--trajectory-root", str(TRAJECTORIES),
        "--object-selection", str(SELECTION), "--output-dir", str(strict_dir),
        "--seed", "2025", "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ])
    with (strict_dir / "screen_summary.yaml").open(encoding="utf-8") as handle:
        strict = yaml.safe_load(handle)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "teacher85 fine-tuning seed robustness",
        "final_unseen_v2_used": False,
        "fixed_initial_checkpoint": str(INIT),
        "changed_variable": "fine-tuning seed 2025 to 2026",
        "coarse_results": coarse_rows,
        "coarse_winner": winner,
        "strict_ranking": strict["ranking"],
    }
    path = OUTPUT / "seed_check_summary.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("TEACHER85_SEED2026_CHECK=COMPLETE summary={}".format(path), flush=True)


if __name__ == "__main__":
    main()
