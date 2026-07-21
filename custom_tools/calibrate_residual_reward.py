"""Calibrate residual-training reward terms without updating a policy."""

import argparse
import copy
import os
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for import_root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from custom_tools import evaluate_bc as evaluation_support  # noqa: E402
from custom_tools.train_residual_ppo import (  # noqa: E402
    DEFAULT_BC_CHECKPOINT, build_task, indexed_trajectory_data,
    REWARD_COMPONENTS)


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Measure each reward term on frozen-BC zero-residual episodes.")
    parser.add_argument("--config", default=str(
        REPO_ROOT / "custom_tools/configs/residual_ppo_smoke.yaml"))
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
    parser.add_argument("--output", default=str(
        REPO_ROOT / "custom_tools/results/reward_calibration/"
        "stage1_zero_residual.yaml"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--show-viewer", action="store_true")
    return parser.parse_args()


def resolve_paths(cli):
    for name in ("config", "trajectory_selection", "trajectory_root",
                 "bc_checkpoint", "bc_config", "env_config", "train_config",
                 "output"):
        setattr(cli, name, str(Path(getattr(cli, name)).expanduser().resolve()))


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def run(cli):
    resolve_paths(cli)
    with open(cli.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(cli.trajectory_selection, "r", encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)
    if selection.get("status") != "frozen_stage1_selection":
        raise ValueError("Stage-1 trajectory selection is not frozen")
    if Path(selection["trajectory_root"]).resolve() != Path(
            cli.trajectory_root).resolve():
        raise ValueError("Calibration and selection trajectory roots differ")

    evaluation_support.initialize_cuda_runtime()
    evaluation_support.require_free_vram(cli.min_free_vram_mb)
    evaluation_support.initialize_runtime()
    import torch
    from custom_tools.residual_env import ResidualDexGraspEnv

    object_ids = selection["object_ids"]
    trajectory_data = indexed_trajectory_data(
        cli.trajectory_root, object_ids,
        selection["trajectory_indices_by_object"])
    cli.num_envs = sum(data["grasp_seqs"].shape[0] for data in trajectory_data)
    original_cwd = Path.cwd()
    task = None
    try:
        os.chdir(str(DEXGRASP_ROOT))
        official_args = evaluation_support.build_official_args(cli)
        base_cfg, cfg_train, _ = evaluation_support.load_cfg(official_args)
        evaluation_support.set_seed(
            cli.seed, cfg_train.get("torch_deterministic", False))
        bc_model, _, checkpoint_path, _ = evaluation_support.load_model(cli)
        task = build_task(
            cli, config, official_args, base_cfg, cfg_train, trajectory_data)
        env = ResidualDexGraspEnv(
            task, bc_model,
            horizon=config["horizon"],
            history_frames=config["history_frames"],
            wrist_residual_scale=config["wrist_residual_scale"],
            finger_residual_scale=config["finger_residual_scale"],
            contact_force_threshold=config["contact_force_threshold"],
            reset_settle_steps=config.get("reset_settle_steps", 4),
            reward_config=config["reward"],
        )
        env.reset()
        num_envs = task.num_envs
        active = torch.ones(num_envs, dtype=torch.bool, device=task.device)
        episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=task.device)
        episode_terms = {
            name: torch.zeros(num_envs, device=task.device)
            for name in ("reward",) + REWARD_COMPONENTS}
        step_sums = {name: 0.0 for name in episode_terms}
        step_squares = {name: 0.0 for name in episode_terms}
        step_absolute = {name: 0.0 for name in episode_terms}
        active_samples = 0
        success = torch.zeros(num_envs, dtype=torch.bool, device=task.device)
        failure = torch.zeros(num_envs, dtype=torch.bool, device=task.device)

        for step in range(int(config["horizon"])):
            zero_residual = torch.zeros((num_envs, 28), device=task.device)
            _, _, _, done, terms = env.step(zero_residual, step + 1)
            active_float = active.float()
            sample_count = int(active.sum().item())
            active_samples += sample_count
            episode_lengths += active.long()
            for name in episode_terms:
                values = terms[name]
                masked = values[active]
                episode_terms[name] += values * active_float
                step_sums[name] += masked.sum().item()
                step_squares[name] += masked.square().sum().item()
                step_absolute[name] += masked.abs().sum().item()
            success |= (terms["success_bonus"] > 0) & active
            failure |= (terms["failure_penalty"] < 0) & active
            active &= ~done
            if not active.any():
                break

        absolute_total = sum(step_absolute[name] for name in REWARD_COMPONENTS)
        per_step = {}
        per_episode = {}
        for name in episode_terms:
            mean = step_sums[name] / active_samples
            variance = max(step_squares[name] / active_samples - mean * mean, 0.0)
            per_step[name] = {
                "mean": mean,
                "std": variance ** 0.5,
                "absolute_mean": step_absolute[name] / active_samples,
            }
            if name in REWARD_COMPONENTS:
                per_step[name]["absolute_contribution_fraction"] = (
                    step_absolute[name] / absolute_total
                    if absolute_total > 0 else 0.0)
            per_episode[name] = distribution(
                episode_terms[name].cpu().numpy())

        env_object_ids = [object_ids[int(index)] for index in task.object_idxs]
        per_object = {}
        for object_id in object_ids:
            indices = [index for index, value in enumerate(env_object_ids)
                       if value == object_id]
            index_tensor = torch.tensor(indices, device=task.device)
            per_object[object_id] = {
                "trajectory_indices": selection[
                    "trajectory_indices_by_object"][object_id],
                "episode_return": distribution(
                    episode_terms["reward"][index_tensor].cpu().numpy()),
                "episode_length": distribution(
                    episode_lengths[index_tensor].cpu().numpy()),
                "official_success_rate": float(
                    success[index_tensor].float().mean().item()),
                "failure_termination_rate": float(
                    failure[index_tensor].float().mean().item()),
                "term_episode_means": {
                    name: float(episode_terms[name][index_tensor].mean().item())
                    for name in REWARD_COMPONENTS},
            }

        component_sum = sum(episode_terms[name] for name in REWARD_COMPONENTS)
        decomposition_error = float(
            (component_sum - episode_terms["reward"]).abs().max().item())
        dominant = [
            name for name in REWARD_COMPONENTS
            if per_step[name]["absolute_contribution_fraction"] > 0.60]
        result = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "frozen_bc_zero_residual_reward_calibration",
            "updates_policy_parameters": False,
            "bc_checkpoint": str(checkpoint_path),
            "trajectory_selection": cli.trajectory_selection,
            "num_envs": num_envs,
            "active_step_samples": active_samples,
            "reward_config": copy.deepcopy(config["reward"]),
            "success_count": int(success.sum().item()),
            "failure_termination_count": int(failure.sum().item()),
            "reward_decomposition_max_error": decomposition_error,
            "dominant_components_over_60_percent": dominant,
            "per_step": per_step,
            "per_episode": per_episode,
            "per_object": per_object,
        }
        output_path = Path(cli.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(result, handle, allow_unicode=True, sort_keys=False)
        print("Reward calibration saved: {}".format(output_path))
        print("success={}/{} failure_termination={}/{} decomposition_error={:.3g}".format(
            int(success.sum().item()), num_envs, int(failure.sum().item()), num_envs,
            decomposition_error))
        for name in REWARD_COMPONENTS:
            stats = per_step[name]
            print("{}: mean={:.4f} std={:.4f} abs_fraction={:.1%}".format(
                name, stats["mean"], stats["std"],
                stats["absolute_contribution_fraction"]))
        return output_path
    finally:
        if task is not None:
            task.clean_sim()
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    run(parse_cli())
