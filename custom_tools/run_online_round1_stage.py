"""Run and validate the complete first online-imitation training stage."""

import argparse
from datetime import datetime
from pathlib import Path
import statistics
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "unified_student_online_r1_seed2025_e20_v1"
RUN_DIR = ROOT / "custom_tools/runs/bc" / RUN_NAME
RESULT_DIR = ROOT / "custom_tools/results/evaluations/online_student_round1_e5_e10_e15_e20"
OFFLINE = (ROOT / "custom_tools/runs/bc/"
           "unified_student_routed_t70_demo30_seed2025_e40_v1/"
           "epoch=039-step=8640.ckpt")
ONLINE_CONFIG = ROOT / "custom_tools/configs/unified_student_online_round1.yaml"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_soup_anchored_gated.yaml"
SELECTION = ROOT / "custom_tools/configs/bc_train16_heldout_trajectory_validation.yaml"
TRAJECTORY_ROOT = ROOT / "dexgrasp/dataset/bc_multicategory_valid"
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
        raise RuntimeError(
            "Expected one epoch {} checkpoint, found {}".format(epoch, matches))
    return matches[0].resolve()


def evaluate(label, checkpoint_path, cli):
    output = RESULT_DIR / (label + "_r1.yaml")
    if output.exists():
        print("REUSE: {}".format(output), flush=True)
        return output
    run([
        sys.executable,
        str(ROOT / "custom_tools/evaluate_bc_checkpoints_isolated.py"),
        "--checkpoint", str(checkpoint_path),
        "--bc-config", str(ONLINE_CONFIG),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORY_ROOT),
        "--object-selection", str(SELECTION),
        "--output", str(output),
        "--seed", "2025",
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ])
    return output


def result(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["checkpoint_results"][0]


def compact(path, repeat):
    item = result(path)
    return {
        "repeat": repeat,
        "output": str(path),
        "success_count": item["total_success_count"],
        "trajectory_count": item["total_trajectory_count"],
        "macro_success_rate": item["macro_official_peak_success_rate"],
        "macro_lift_m": item["macro_mean_maximum_lift_m"],
        "macro_failure_rate": item["macro_failure_rate"],
    }


def main():
    cli = parse_cli()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    expected = [RUN_DIR / "last.ckpt"]
    if not all(path.is_file() for path in expected):
        if RUN_DIR.exists() and list(RUN_DIR.glob("*.ckpt")):
            raise RuntimeError(
                "Partial training run exists; inspect it before resuming: {}".format(RUN_DIR))
        run([
            sys.executable, str(ROOT / "custom_tools/train_bc.py"),
            "--config", str(ONLINE_CONFIG),
            "--run-name", RUN_NAME,
            "--init-checkpoint", str(OFFLINE),
            "--min-free-vram-mb", str(cli.min_free_vram_mb),
        ])
    else:
        print("REUSE completed training: {}".format(RUN_DIR), flush=True)

    candidates = {"online_e{:02d}".format(epoch): checkpoint(epoch)
                  for epoch in EPOCHS}
    first_outputs = {
        label: evaluate(label, path, cli) for label, path in candidates.items()}
    first_outputs["offline_control"] = evaluate(
        "offline_control", OFFLINE.resolve(), cli)

    ranked = sorted(candidates, key=lambda label: (
        -result(first_outputs[label])["total_success_count"],
        -result(first_outputs[label])["macro_official_peak_success_rate"],
        -result(first_outputs[label])["macro_mean_maximum_lift_m"],
        result(first_outputs[label])["macro_failure_rate"],
        label,
    ))
    winner = ranked[0]
    repeats_dir = RESULT_DIR / "winner_vs_offline_repeats"
    run([
        sys.executable, str(ROOT / "custom_tools/repeat_strict_bc_finalists.py"),
        "--candidate", "online_best={}".format(candidates[winner]),
        "--candidate", "offline_control={}".format(OFFLINE.resolve()),
        "--repeat-start", "2", "--repeat-end", "3",
        "--bc-config", str(ONLINE_CONFIG),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORY_ROOT),
        "--object-selection", str(SELECTION),
        "--output-dir", str(repeats_dir),
        "--seed", "2025",
        "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ])

    repeated = {
        "online_best": [compact(first_outputs[winner], 1)],
        "offline_control": [compact(first_outputs["offline_control"], 1)],
    }
    for label in repeated:
        for repeat in (2, 3):
            repeated[label].append(compact(
                repeats_dir / ("{}_r{}.yaml".format(label, repeat)), repeat))
    aggregate = {}
    for label, runs in repeated.items():
        counts = [item["success_count"] for item in runs]
        macros = [item["macro_success_rate"] for item in runs]
        aggregate[label] = {
            "runs": runs,
            "mean_success_count": statistics.mean(counts),
            "success_count_population_std": statistics.pstdev(counts),
            "mean_macro_success_rate": statistics.mean(macros),
            "mean_macro_lift_m": statistics.mean(
                item["macro_lift_m"] for item in runs),
            "mean_macro_failure_rate": statistics.mean(
                item["macro_failure_rate"] for item in runs),
        }
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "offline student plus one online imitation aggregation round",
        "selection_rule": (
            "official success count, macro success, macro lift, failure rate"),
        "final_unseen_v2_used": False,
        "single_run_candidates": {
            label: compact(path, 1) for label, path in first_outputs.items()},
        "selected_online_checkpoint": winner,
        "selected_online_checkpoint_path": str(candidates[winner]),
        "three_repeat_comparison": aggregate,
    }
    summary_path = RESULT_DIR / "stage_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("ONLINE_ROUND1_STAGE=COMPLETE summary={}".format(summary_path), flush=True)


if __name__ == "__main__":
    main()
