"""Evaluate a residual policy while preserving the official success flag."""

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
from datetime import datetime

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for import_root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from custom_tools import evaluate_bc as evaluation_support  # noqa: E402
from custom_tools.train_residual_ppo import (  # noqa: E402
    DEFAULT_BC_CHECKPOINT, DEFAULT_MANIFEST, DEFAULT_TRAJECTORY_ROOT,
    build_task, indexed_trajectory_data, sliced_trajectory_data)


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Official-metric evaluation for BC plus residual PPO.")
    parser.add_argument("--object-id", action="append", default=[])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--manifest-split", choices=("train", "test"), default="train")
    parser.add_argument("--trajectory-root", default=str(DEFAULT_TRAJECTORY_ROOT))
    parser.add_argument("--trajectory-start", type=int, default=0)
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument(
        "--trajectory-indices", default="",
        help="Comma-separated source indices; overrides contiguous slicing.")
    parser.add_argument("--residual-checkpoint", default="")
    parser.add_argument("--zero-residual", action="store_true")
    parser.add_argument("--residual-config", default=str(
        REPO_ROOT / "custom_tools/configs/residual_ppo_smoke.yaml"))
    parser.add_argument("--bc-checkpoint", default=str(DEFAULT_BC_CHECKPOINT))
    parser.add_argument("--bc-config", default=str(
        REPO_ROOT / "custom_tools/configs/multicategory_bc_formal.yaml"))
    parser.add_argument("--env-config", default=str(
        DEXGRASP_ROOT / "cfg/shadow_hand_grasp_dexrep_ijrr.yaml"))
    parser.add_argument("--train-config", default=str(
        DEXGRASP_ROOT / "cfg/ppo1/config.yaml"))
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--capture-dir", default="")
    parser.add_argument("--capture-env", type=int, default=0)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-stride", type=int, default=2)
    return parser.parse_args()


def resolve_paths(cli):
    for name in ("manifest", "trajectory_root", "residual_checkpoint",
                 "residual_config", "bc_checkpoint", "bc_config",
                 "env_config", "train_config", "output", "capture_dir"):
        value = getattr(cli, name)
        if value:
            setattr(cli, name, str(Path(value).expanduser().resolve()))


def expand_objects(cli):
    if cli.object_id:
        return list(dict.fromkeys(cli.object_id))
    with open(cli.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    objects = []
    for category in manifest["criteria"]["categories"]:
        objects.extend(manifest["categories"][category][cli.manifest_split])
    return objects


def load_residual_model(checkpoint_path, device):
    import torch
    from custom_tools.residual_ppo import ResidualActorCritic
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ResidualActorCritic(
        checkpoint["actor_obs_dim"], checkpoint["critic_obs_dim"],
        hidden_dims=tuple(config["hidden_dims"]), init_std=config["init_std"],
        gate_dim=2 if config.get("gated_residual", False) else 0,
        initial_gate=config.get("initial_gate", 0.1))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def run(cli):
    if cli.zero_residual == bool(cli.residual_checkpoint):
        raise ValueError(
            "Choose exactly one: --zero-residual or --residual-checkpoint PATH")
    if cli.num_trajectories < 0:
        raise ValueError("--num-trajectories cannot be negative; use 0 for all")
    if cli.capture_stride < 1:
        raise ValueError("--capture-stride must be positive")
    trajectory_indices = None
    if cli.trajectory_indices:
        trajectory_indices = [
            int(item) for item in cli.trajectory_indices.split(",") if item]
        if not trajectory_indices:
            raise ValueError("--trajectory-indices cannot be empty")
        if len(set(trajectory_indices)) != len(trajectory_indices):
            raise ValueError("--trajectory-indices cannot contain duplicates")
        if cli.trajectory_start != 0:
            raise ValueError(
                "--trajectory-start cannot be combined with explicit indices")
    resolve_paths(cli)
    object_ids = expand_objects(cli)
    with open(cli.residual_config, "r", encoding="utf-8") as handle:
        residual_config = yaml.safe_load(handle)

    evaluation_support.initialize_cuda_runtime()
    evaluation_support.require_free_vram(cli.min_free_vram_mb)
    evaluation_support.initialize_runtime()
    import torch
    from custom_tools.residual_env import ResidualDexGraspEnv

    original_cwd = Path.cwd()
    residual_model = None
    residual_checkpoint = None
    if cli.residual_checkpoint:
        residual_model, residual_checkpoint = load_residual_model(
            cli.residual_checkpoint, cli.rl_device)
    results = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "zero_residual_bc_control" if cli.zero_residual
                else "deterministic_residual_policy",
        "bc_checkpoint": cli.bc_checkpoint,
        "residual_checkpoint": cli.residual_checkpoint or None,
        "residual_iteration": (
            residual_checkpoint.get("iteration") if residual_checkpoint else None),
        "seed": cli.seed,
        "trajectory_root": cli.trajectory_root,
        "trajectory_start": cli.trajectory_start,
        "num_trajectories": cli.num_trajectories,
        "trajectory_indices": trajectory_indices,
        "success_metric": "official_peak",
        "official_success_definition_changed": False,
        "capture_dir": cli.capture_dir or None,
        "objects": [],
    }
    try:
        os.chdir(str(DEXGRASP_ROOT))
        cli.num_envs = max(
            1, len(trajectory_indices) if trajectory_indices is not None
            else cli.num_trajectories)
        official_args = evaluation_support.build_official_args(cli)
        base_cfg, cfg_train, _ = evaluation_support.load_cfg(official_args)
        evaluation_support.set_seed(
            cli.seed, cfg_train.get("torch_deterministic", False))
        bc_model, _, checkpoint_path, _ = evaluation_support.load_model(cli)
        results["bc_checkpoint"] = str(checkpoint_path)
        total_successes = 0
        total_trajectories = 0

        for object_id in object_ids:
            task = None
            try:
                trajectory_path = Path(cli.trajectory_root) / "{}.npy".format(object_id)
                if trajectory_indices is not None:
                    evaluated_indices = list(trajectory_indices)
                    trajectory_count = len(evaluated_indices)
                    trajectory_data = indexed_trajectory_data(
                        cli.trajectory_root, [object_id],
                        {object_id: evaluated_indices})
                elif cli.num_trajectories == 0:
                    import numpy as np
                    source = np.load(trajectory_path, allow_pickle=True).item()
                    trajectory_count = (
                        source["grasp_seqs"].shape[0] - cli.trajectory_start)
                    evaluated_indices = list(range(
                        cli.trajectory_start,
                        cli.trajectory_start + trajectory_count))
                    trajectory_data = sliced_trajectory_data(
                        cli.trajectory_root, [object_id], cli.trajectory_start,
                        trajectory_count)
                else:
                    trajectory_count = cli.num_trajectories
                    evaluated_indices = list(range(
                        cli.trajectory_start,
                        cli.trajectory_start + trajectory_count))
                    trajectory_data = sliced_trajectory_data(
                        cli.trajectory_root, [object_id], cli.trajectory_start,
                        trajectory_count)
                per_object_config = copy.deepcopy(residual_config)
                per_object_config["seed"] = cli.seed
                task = build_task(
                    cli, per_object_config, official_args, base_cfg, cfg_train,
                    trajectory_data)
                env = ResidualDexGraspEnv(
                    task, bc_model,
                    horizon=per_object_config["horizon"],
                    history_frames=per_object_config["history_frames"],
                    wrist_residual_scale=per_object_config["wrist_residual_scale"],
                    finger_residual_scale=per_object_config["finger_residual_scale"],
                    contact_force_threshold=per_object_config["contact_force_threshold"],
                    reset_settle_steps=per_object_config.get(
                        "reset_settle_steps", 4),
                    reward_config=per_object_config.get("reward"),
                    gated_residual=per_object_config.get(
                        "gated_residual", False),
                )
                actor_obs, critic_obs = env.reset()
                if residual_model is not None:
                    checkpoint_gated = bool(
                        residual_checkpoint["config"].get(
                            "gated_residual", False))
                    if checkpoint_gated != env.gated_residual:
                        raise ValueError(
                            "Residual config gated_residual does not match checkpoint")
                    if (env.actor_obs_dim != residual_checkpoint["actor_obs_dim"]
                            or env.critic_obs_dim != residual_checkpoint["critic_obs_dim"]):
                        raise ValueError("Residual checkpoint observation dimensions mismatch")
                peak_count = 0
                peak_mask = torch.zeros(
                    task.num_envs, dtype=torch.bool, device=task.device)
                ever_success = torch.zeros(
                    task.num_envs, dtype=torch.bool, device=task.device)
                ever_failure = torch.zeros(
                    task.num_envs, dtype=torch.bool, device=task.device)
                max_height = torch.full(
                    (task.num_envs,), -float("inf"), device=task.device)
                for step in range(int(per_object_config["horizon"])):
                    if cli.zero_residual:
                        action = torch.zeros(
                            (task.num_envs, env.policy_action_dim),
                            device=task.device)
                    else:
                        with torch.no_grad():
                            action, _, _ = residual_model.act(
                                actor_obs, critic_obs, deterministic=True)
                    actor_obs, critic_obs, _, _, terms = env.step(action, step + 1)
                    if cli.capture_dir:
                        task.capture_frame(step)
                    success = task.successes > 0
                    ever_success |= success
                    ever_failure |= terms["failure_penalty"] < 0
                    current_count = int(success.sum().item())
                    if current_count > peak_count:
                        peak_count = current_count
                        peak_mask = success.clone()
                    max_height = torch.maximum(max_height, terms["height_delta"])
                object_result = {
                    "object_id": object_id,
                    "trajectory_count": task.num_envs,
                    "trajectory_indices": evaluated_indices,
                    "official_peak_success_count": peak_count,
                    "official_peak_success_rate": peak_count / task.num_envs,
                    "official_peak_success_indices": peak_mask.nonzero(
                        as_tuple=False).flatten().cpu().tolist(),
                    "official_peak_success_source_indices": [
                        evaluated_indices[index]
                        for index in peak_mask.nonzero(
                            as_tuple=False).flatten().cpu().tolist()],
                    "diagnostic_ever_success_count": int(ever_success.sum().item()),
                    "diagnostic_ever_success_rate": float(ever_success.float().mean().item()),
                    "diagnostic_ever_success_indices": ever_success.nonzero(
                        as_tuple=False).flatten().cpu().tolist(),
                    "diagnostic_failure_rate": float(
                        ever_failure.float().mean().item()),
                    "diagnostic_mean_maximum_lift_m": float(max_height.mean().item()),
                    "diagnostic_maximum_lift_m_by_trajectory": [
                        float(value) for value in max_height.cpu().tolist()],
                }
                results["objects"].append(object_result)
                total_successes += peak_count
                total_trajectories += task.num_envs
                print("{} official_peak={}/{} ({:.2f}%)".format(
                    object_id, peak_count, task.num_envs,
                    100.0 * object_result["official_peak_success_rate"]))
            finally:
                if task is not None:
                    task.clean_sim()
                gc.collect()
                torch.cuda.empty_cache()

        results["overall_official_peak_success_rate"] = (
            total_successes / total_trajectories)
        if cli.output:
            output = Path(cli.output)
        else:
            tag = "zero_residual" if cli.zero_residual else "residual_policy"
            output = REPO_ROOT / "custom_tools/results/evaluations" / (
                "{}_seed{}_{}.yaml".format(
                    tag, cli.seed, datetime.now().strftime("%Y%m%d_%H%M%S")))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(results, handle, allow_unicode=True, sort_keys=False)
        print("Saved evaluation: {}".format(output))
        return output
    finally:
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    run(parse_cli())
