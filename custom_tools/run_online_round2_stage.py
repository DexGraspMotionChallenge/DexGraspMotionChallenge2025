"""Merge round data, train round two, and run strict repeated evaluation."""

import argparse
from datetime import datetime
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "unified_student_online_r2_seed2025_e10_v1"
RUN_DIR = ROOT / "custom_tools/runs/bc" / RUN_NAME
RESULT_DIR = ROOT / "custom_tools/results/evaluations/online_student_round2_e2_e4_e6_e8_e10"
CONTROL = (ROOT / "custom_tools/runs/bc/unified_student_online_r1_seed2025_e20_v1/"
           "epoch=014-step=6420.ckpt")
CONFIG = ROOT / "custom_tools/configs/unified_student_online_round2.yaml"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_soup_anchored_gated.yaml"
SELECTION = ROOT / "custom_tools/configs/bc_train16_heldout_trajectory_validation.yaml"
TRAJECTORY_ROOT = ROOT / "dexgrasp/dataset/bc_multicategory_valid"
MERGED = ROOT / "custom_tools/data/distillation/online_round12_bounded.npz"
EPOCHS = (2, 4, 6, 8, 10)


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
        raise RuntimeError("Expected one epoch {} checkpoint: {}".format(epoch, matches))
    return matches[0].resolve()


def load_result(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["checkpoint_results"][0]


def compact(path, repeat):
    item = load_result(path)
    return {
        "repeat": repeat, "output": str(path),
        "success_count": item["total_success_count"],
        "trajectory_count": item["total_trajectory_count"],
        "macro_success_rate": item["macro_official_peak_success_rate"],
        "macro_lift_m": item["macro_mean_maximum_lift_m"],
        "macro_failure_rate": item["macro_failure_rate"],
    }


def evaluate(label, checkpoint_path, cli):
    output = RESULT_DIR / (label + "_r1.yaml")
    if output.exists():
        print("REUSE: {}".format(output), flush=True)
        return output
    run([
        sys.executable, str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint", str(checkpoint_path), "--bc-config", str(CONFIG),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORY_ROOT),
        "--object-selection", str(SELECTION), "--output", str(output),
        "--seed", "2025", "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ])
    return output


def main():
    cli = parse_cli()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if not MERGED.exists():
        run([
            sys.executable, str(ROOT / "custom_tools/merge_online_imitation_rounds.py"),
            "--round1", str(ROOT / "custom_tools/data/distillation/online_round1_train.npz"),
            "--round2", str(ROOT / "custom_tools/data/distillation/online_round2_train.npz"),
            "--output", str(MERGED), "--seed", "2025",
        ])
    if not (RUN_DIR / "last.ckpt").is_file():
        if RUN_DIR.exists() and list(RUN_DIR.glob("*.ckpt")):
            raise RuntimeError("Partial training run exists: {}".format(RUN_DIR))
        run([
            sys.executable, str(ROOT / "custom_tools/train_bc.py"),
            "--config", str(CONFIG), "--run-name", RUN_NAME,
            "--init-checkpoint", str(CONTROL),
            "--min-free-vram-mb", str(cli.min_free_vram_mb),
        ])
    candidates = {"online_r2_e{:02d}".format(e): checkpoint(e) for e in EPOCHS}
    outputs = {label: evaluate(label, path, cli) for label, path in candidates.items()}
    outputs["round1_control"] = evaluate("round1_control", CONTROL.resolve(), cli)
    ranked = sorted(candidates, key=lambda label: (
        -load_result(outputs[label])["total_success_count"],
        -load_result(outputs[label])["macro_official_peak_success_rate"],
        load_result(outputs[label])["macro_failure_rate"],
        -load_result(outputs[label])["macro_mean_maximum_lift_m"], label))
    winner = ranked[0]
    repeats = RESULT_DIR / "winner_vs_round1_repeats"
    run([
        sys.executable, str(ROOT / "custom_tools/repeat_strict_bc_finalists.py"),
        "--candidate", "round2_best={}".format(candidates[winner]),
        "--candidate", "round1_control={}".format(CONTROL.resolve()),
        "--repeat-start", "2", "--repeat-end", "3",
        "--bc-config", str(CONFIG), "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORY_ROOT),
        "--object-selection", str(SELECTION), "--output-dir", str(repeats),
        "--seed", "2025", "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ])
    groups = {
        "round2_best": [compact(outputs[winner], 1)],
        "round1_control": [compact(outputs["round1_control"], 1)],
    }
    for label in groups:
        groups[label].extend(compact(
            repeats / "{}_r{}.yaml".format(label, repeat), repeat)
            for repeat in (2, 3))
    aggregate = {}
    for label, runs in groups.items():
        counts = [x["success_count"] for x in runs]
        aggregate[label] = {
            "runs": runs, "mean_success_count": statistics.mean(counts),
            "success_count_population_std": statistics.pstdev(counts),
            "mean_macro_success_rate": statistics.mean(x["macro_success_rate"] for x in runs),
            "mean_macro_lift_m": statistics.mean(x["macro_lift_m"] for x in runs),
            "mean_macro_failure_rate": statistics.mean(x["macro_failure_rate"] for x in runs),
        }
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "second conservative online imitation round",
        "selection_rule": "success count, macro success, failure rate, lift",
        "final_unseen_v2_used": False,
        "single_run_candidates": {label: compact(path, 1) for label, path in outputs.items()},
        "selected_round2_checkpoint": winner,
        "selected_round2_checkpoint_path": str(candidates[winner]),
        "three_repeat_comparison": aggregate,
    }
    path = RESULT_DIR / "stage_summary.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("ONLINE_ROUND2_STAGE=COMPLETE summary={}".format(path), flush=True)


if __name__ == "__main__":
    main()
