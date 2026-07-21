"""Run the locked report-only final evaluation on unseen-v2."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "custom_tools/results/evaluations/final_unseen_v2_locked"
MANIFEST = ROOT / "custom_tools/configs/unseen_v2_final.json"
SELECTION = ROOT / "custom_tools/configs/unseen_v2_final_selection.yaml"
TRAJECTORIES = ROOT / "dexgrasp/dataset/unseen_v2_final"
BC_CONFIG = ROOT / "custom_tools/configs/unified_student_online_round1.yaml"
RESIDUAL_CONFIG = ROOT / "custom_tools/configs/residual_ppo_soup_anchored_gated.yaml"
SOUP = ROOT / "custom_tools/runs/bc/model_soups/noise005_s2025_s2026_weighted2to1.ckpt"
UNIFIED = (ROOT / "custom_tools/runs/bc/unified_hparam_teacher85_seed2025_e20_v1/"
           "epoch=004-step=2140.ckpt")
TEACHERS = {
    "bottle": ROOT / "custom_tools/runs/bc/category_expert_bottle_noise005_soup_seed2025_e40_v1/epoch=039-step=2560.ckpt",
    "mug": SOUP,
    "bowl": ROOT / "custom_tools/runs/bc/category_expert_bowl_noise005_soup_seed2025_e40_v1/epoch=039-step=1680.ckpt",
    "camera": ROOT / "custom_tools/runs/bc/category_expert_camera_noise005_soup_seed2025_e40_v1/epoch=009-step=500.ckpt",
}


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def run(command):
    print("RUN: {}".format(" ".join(str(x) for x in command)), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def main():
    cli = parse_cli()
    with MANIFEST.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "frozen_before_any_learned_policy_evaluation":
        raise ValueError("Unseen-v2 manifest is not in the frozen state")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    single_dir = OUTPUT / "single_networks"
    routed_dir = OUTPUT / "routed_teacher_pool"
    run([
        sys.executable, str(ROOT / "custom_tools/repeat_strict_bc_finalists.py"),
        "--candidate", "soup_baseline={}".format(SOUP),
        "--candidate", "unified_online_teacher85={}".format(UNIFIED),
        "--repeat-start", "1", "--repeat-end", "3",
        "--bc-config", str(BC_CONFIG), "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORIES),
        "--object-selection", str(SELECTION), "--output-dir", str(single_dir),
        "--seed", "2025", "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ])
    routed_command = [
        sys.executable, str(ROOT / "custom_tools/evaluate_routed_bc_repeats.py")]
    for category, checkpoint in TEACHERS.items():
        routed_command.extend(["--teacher", "{}={}".format(category, checkpoint)])
    routed_command.extend([
        "--repeats", "3", "--manifest", str(MANIFEST),
        "--manifest-split", "test", "--bc-config", str(BC_CONFIG),
        "--residual-config", str(RESIDUAL_CONFIG),
        "--trajectory-root", str(TRAJECTORIES), "--output-dir", str(routed_dir),
        "--seed", "2025", "--min-free-vram-mb", str(cli.min_free_vram_mb),
        "--max-attempts", str(cli.max_attempts),
    ])
    run(routed_command)
    with (single_dir / "repeats_summary.yaml").open(encoding="utf-8") as handle:
        single = yaml.safe_load(handle)
    with (routed_dir / "routed_repeats_summary.yaml").open(encoding="utf-8") as handle:
        routed = yaml.safe_load(handle)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "locked_final_unseen_v2_report_only",
        "post_evaluation_training_allowed": False,
        "manifest": str(MANIFEST),
        "objects": manifest["categories"],
        "single_network_results": single["candidates"],
        "routed_teacher_pool_results": routed["aggregate"],
        "routed_teacher_pool_repeats": routed["repeats"],
    }
    summary_path = OUTPUT / "final_summary.yaml"
    with summary_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
    print("FINAL_UNSEEN_V2=COMPLETE summary={}".format(summary_path), flush=True)


if __name__ == "__main__":
    main()
