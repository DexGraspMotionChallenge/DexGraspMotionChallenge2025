"""Diagnose zero-residual transitions immediately after indexed PPO resets."""

import argparse
import os
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for import_root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from custom_tools import evaluate_bc as evaluation_support  # noqa: E402
from custom_tools.train_residual_ppo import (  # noqa: E402
    DEFAULT_BC_CHECKPOINT, build_task, indexed_trajectory_data)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(
        REPO_ROOT / "custom_tools/configs/residual_ppo_stage1.yaml"))
    parser.add_argument("--trajectory-selection", default=str(
        REPO_ROOT / "custom_tools/configs/residual_stage1_trajectory_selection.yaml"))
    parser.add_argument("--trajectory-root", default=str(
        DEXGRASP_ROOT / "dataset/bc_multicategory_train"))
    parser.add_argument("--bc-checkpoint", default=str(DEFAULT_BC_CHECKPOINT))
    parser.add_argument("--bc-config", default=str(
        REPO_ROOT / "custom_tools/configs/multicategory_bc_formal.yaml"))
    parser.add_argument("--env-config", default=str(
        DEXGRASP_ROOT / "cfg/shadow_hand_grasp_dexrep_ijrr.yaml"))
    parser.add_argument("--train-config", default=str(
        DEXGRASP_ROOT / "cfg/ppo1/config.yaml"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--show-viewer", action="store_true")
    return parser.parse_args()


def run(cli):
    for name in ("config", "trajectory_selection", "trajectory_root",
                 "bc_checkpoint", "bc_config", "env_config", "train_config"):
        setattr(cli, name, str(Path(getattr(cli, name)).resolve()))
    with open(cli.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(cli.trajectory_selection, "r", encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)

    evaluation_support.initialize_cuda_runtime()
    evaluation_support.require_free_vram(cli.min_free_vram_mb)
    evaluation_support.initialize_runtime()
    import torch
    from custom_tools.residual_env import ResidualDexGraspEnv

    object_ids = selection["object_ids"]
    trajectory_data = indexed_trajectory_data(
        cli.trajectory_root, object_ids,
        selection["trajectory_indices_by_object"])
    cli.num_envs = sum(item["grasp_seqs"].shape[0] for item in trajectory_data)
    original_cwd = Path.cwd()
    task = None
    try:
        os.chdir(str(DEXGRASP_ROOT))
        official_args = evaluation_support.build_official_args(cli)
        base_cfg, cfg_train, _ = evaluation_support.load_cfg(official_args)
        evaluation_support.set_seed(cli.seed, False)
        bc_model, _, _, _ = evaluation_support.load_model(cli)
        task = build_task(
            cli, config, official_args, base_cfg, cfg_train, trajectory_data)
        env = ResidualDexGraspEnv(
            task, bc_model, horizon=config["horizon"],
            history_frames=config["history_frames"],
            wrist_residual_scale=config["wrist_residual_scale"],
            finger_residual_scale=config["finger_residual_scale"],
            contact_force_threshold=config["contact_force_threshold"],
            reset_settle_steps=config.get("reset_settle_steps", 4),
            reward_config=config["reward"])
        env.reset()
        zero = torch.zeros((task.num_envs, 28), device=task.device)
        for step in range(1, cli.steps + 1):
            _, _, _, done, terms = env.step(zero, step)
            failures = int((terms["failure_penalty"] < 0).sum().item())
            if step >= config["horizon"] - 2 or failures or done.any():
                print(
                    "step={} progress=[{},{}] done={} failures={} "
                    "height_delta_mean={:.4f} height_delta_min={:.4f}".format(
                        step, int(task.progress_buf.min().item()),
                        int(task.progress_buf.max().item()), int(done.sum().item()),
                        failures, float(terms["height_delta"].mean().item()),
                        float(terms["height_delta"].min().item())))
            if done.any():
                env.reset_done(done)
    finally:
        if task is not None:
            task.clean_sim()
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    run(parse_cli())
