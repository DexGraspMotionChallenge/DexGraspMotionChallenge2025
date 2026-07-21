"""Train a shared bounded residual PPO policy on top of a frozen BC policy."""

import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for import_root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from custom_tools import evaluate_bc as evaluation_support  # noqa: E402


DEFAULT_BC_CHECKPOINT = (
    REPO_ROOT / "custom_tools/runs/bc/"
    "multicategory_bc_warmstart_seed2025_e100/last.ckpt")
DEFAULT_MANIFEST = REPO_ROOT / "custom_tools/configs/object_split_final.json"
DEFAULT_TRAJECTORY_ROOT = DEXGRASP_ROOT / "dataset/bc_multicategory_train"


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Frozen-BC plus shared bounded residual PPO trainer.")
    parser.add_argument("--config", default=str(
        REPO_ROOT / "custom_tools/configs/residual_ppo_smoke.yaml"))
    parser.add_argument("--bc-checkpoint", default=str(DEFAULT_BC_CHECKPOINT))
    parser.add_argument("--bc-config", default=str(
        REPO_ROOT / "custom_tools/configs/multicategory_bc_formal.yaml"))
    parser.add_argument("--env-config", default=str(
        DEXGRASP_ROOT / "cfg/shadow_hand_grasp_dexrep_ijrr.yaml"))
    parser.add_argument("--train-config", default=str(
        DEXGRASP_ROOT / "cfg/ppo1/config.yaml"))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--trajectory-root", default=str(DEFAULT_TRAJECTORY_ROOT))
    parser.add_argument("--trajectory-selection", default="")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--history-frames", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--validation-patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def absolute_paths(cli):
    for name in ("config", "bc_checkpoint", "bc_config", "env_config",
                 "train_config", "manifest", "trajectory_root",
                 "trajectory_selection", "resume_checkpoint"):
        value = getattr(cli, name)
        if value:
            setattr(cli, name, str(Path(value).expanduser().resolve()))


def load_run_config(cli):
    with open(cli.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for cli_name, config_name in (
            ("iterations", "iterations"), ("rollout_steps", "rollout_steps"),
            ("history_frames", "history_frames"), ("seed", "seed")):
        value = getattr(cli, cli_name)
        if value is not None:
            config[config_name] = value
    if cli.validation_interval is not None:
        config.setdefault("validation", {})["interval"] = cli.validation_interval
    if cli.validation_patience is not None:
        config.setdefault("validation", {})["patience"] = cli.validation_patience
    return config


def selected_objects(manifest_path, objects_per_category):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    selected = []
    for category in manifest["criteria"]["categories"]:
        category_objects = manifest["categories"][category]["train"]
        selected.extend(category_objects[:objects_per_category])
    return selected


def sliced_trajectory_data(root, object_ids, trajectory_index, count):
    data_list = []
    for object_id in object_ids:
        path = Path(root) / "{}.npy".format(object_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        source = np.load(path, allow_pickle=True).item()
        stop = trajectory_index + count
        available = source["grasp_seqs"].shape[0]
        if stop > available:
            raise IndexError(
                "{} asks for trajectories [{}:{}) but only {} are available"
                .format(object_id, trajectory_index, stop, available))
        data = {"obj_code": object_id}
        for key in ("obj_scale", "obj_rotmat", "grasp_seqs"):
            data[key] = source[key][trajectory_index:stop].copy()
        data_list.append(data)
    return data_list


def indexed_trajectory_data(root, object_ids, indices_by_object):
    data_list = []
    for object_id in object_ids:
        path = Path(root) / "{}.npy".format(object_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        source = np.load(path, allow_pickle=True).item()
        indices = [int(index) for index in indices_by_object[object_id]]
        available = source["grasp_seqs"].shape[0]
        invalid = [index for index in indices if index < 0 or index >= available]
        if invalid:
            raise IndexError("{} has invalid indices {}".format(object_id, invalid))
        data = {"obj_code": object_id}
        for key in ("obj_scale", "obj_rotmat", "grasp_seqs"):
            data[key] = source[key][indices].copy()
        data_list.append(data)
    return data_list


def build_task(cli, config, official_args, base_cfg, cfg_train, trajectory_data):
    cfg = copy.deepcopy(base_cfg)
    object_ids = [data["obj_code"] for data in trajectory_data]
    total_envs = sum(data["grasp_seqs"].shape[0] for data in trajectory_data)
    cfg["seed"] = config["seed"]
    cfg["env"]["seed"] = config["seed"]
    cfg["env"]["numEnvs"] = total_envs
    cfg["env"]["env_mode"] = "bc_env_infer"
    cfg["env"]["observationType"] = "DexRep"
    cfg["env"]["obj_type"] = "one"
    cfg["env"]["object_code_dict"] = object_ids
    cfg["env"]["trajectory_indices"] = []
    cfg["env"]["diagnostic_dense_reward"] = False
    cfg["env"]["evaluation_step_logging"] = False
    cfg["env"]["capture_dir"] = str(getattr(cli, "capture_dir", "") or "")
    cfg["env"]["capture_env"] = int(getattr(cli, "capture_env", 0))
    cfg["env"]["capture_width"] = int(getattr(cli, "capture_width", 640))
    cfg["env"]["capture_height"] = int(getattr(cli, "capture_height", 480))
    cfg["env"]["capture_stride"] = int(getattr(cli, "capture_stride", 1))
    cfg["env"].setdefault("seq_start_rot_uniform", False)
    sim_params = evaluation_support.parse_sim_params(
        official_args, cfg, cfg_train)
    return evaluation_support.CustomShadowHandGraspDexRepIjrr(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=official_args.physics_engine,
        device_type=official_args.device,
        device_id=official_args.device_id,
        headless=official_args.headless,
        is_multi_agent=False,
        npy_list=trajectory_data,
    )


REWARD_COMPONENTS = (
    "approach", "contact", "lift", "milestone", "success_bonus",
    "failure_penalty", "residual_penalty", "smoothness_penalty",
    "gate_penalty")


def reward_term_statistics(term_sums, term_squares, term_absolute_sums,
                           sample_count):
    metrics = {}
    absolute_total = sum(term_absolute_sums.get(name, 0.0)
                         for name in REWARD_COMPONENTS)
    for name, value in term_sums.items():
        mean = value / sample_count
        variance = max(term_squares[name] / sample_count - mean * mean, 0.0)
        metrics["reward_{}_mean".format(name)] = mean
        metrics["reward_{}_std".format(name)] = variance ** 0.5
        if name in REWARD_COMPONENTS:
            metrics["reward_{}_absolute_fraction".format(name)] = (
                term_absolute_sums[name] / absolute_total
                if absolute_total > 0 else 0.0)
    return metrics


def save_checkpoint(path, model, optimizer, config, iteration, global_step,
                    actor_obs_dim, critic_obs_dim, object_ids,
                    validation_metrics=None):
    import torch
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "iteration": iteration,
        "global_step": global_step,
        "actor_obs_dim": actor_obs_dim,
        "critic_obs_dim": critic_obs_dim,
        "object_ids": object_ids,
        "validation_metrics": validation_metrics,
    }, path)


def validation_is_better(current, best, lift_tolerance_m=1e-3):
    """Lexicographic checkpoint rule fixed before formal training."""
    if best is None:
        return True
    success_delta = (
        current["macro_official_peak_success_rate"]
        - best["macro_official_peak_success_rate"])
    if success_delta > 1e-12:
        return True
    if abs(success_delta) > 1e-12:
        return False
    lift_delta = current["macro_mean_maximum_lift_m"] - best[
        "macro_mean_maximum_lift_m"]
    if lift_delta > lift_tolerance_m:
        return True
    if abs(lift_delta) > lift_tolerance_m:
        return False
    return current["macro_failure_rate"] < best["macro_failure_rate"] - 1e-12


def create_residual_env(task, bc_model, config, residual_env_class):
    env = residual_env_class(
        task, bc_model,
        horizon=config["horizon"],
        history_frames=config["history_frames"],
        wrist_residual_scale=config["wrist_residual_scale"],
        finger_residual_scale=config["finger_residual_scale"],
        contact_force_threshold=config["contact_force_threshold"],
        reset_settle_steps=config.get("reset_settle_steps", 4),
        reward_config=config.get("reward"),
        gated_residual=config.get("gated_residual", False),
    )
    actor_obs, critic_obs = env.reset()
    return env, actor_obs, critic_obs


def category_context(task, object_ids, torch):
    env_categories = [
        object_ids[int(object_index)].split("-", 2)[1]
        for object_index in task.object_idxs]
    category_names = sorted(set(env_categories))
    category_masks = {
        category: torch.tensor(
            [item == category for item in env_categories],
            dtype=torch.bool, device=task.device)
        for category in category_names}
    category_to_index = {
        category: index for index, category in enumerate(category_names)}
    advantage_groups = torch.tensor(
        [category_to_index[category] for category in env_categories],
        dtype=torch.long, device=task.device)
    return category_names, category_masks, advantage_groups


def evaluate_validation_policy(cli, config, validation_config, model, bc_model,
                               official_args, base_cfg, cfg_train,
                               residual_env_class, torch):
    """Evaluate held-out same-object trajectories without gradient updates."""
    validation_root = Path(validation_config["trajectory_root"])
    object_ids = validation_config["object_ids"]
    per_object = []
    model.eval()
    for object_id in object_ids:
        task = None
        try:
            source = np.load(
                validation_root / "{}.npy".format(object_id),
                allow_pickle=True).item()
            trajectory_count = source["grasp_seqs"].shape[0]
            trajectory_data = sliced_trajectory_data(
                validation_root, [object_id], 0, trajectory_count)
            task = build_task(
                cli, config, official_args, base_cfg, cfg_train,
                trajectory_data)
            env, actor_obs, critic_obs = create_residual_env(
                task, bc_model, config, residual_env_class)
            peak_count = 0
            max_height = torch.full(
                (task.num_envs,), -float("inf"), device=task.device)
            ever_failure = torch.zeros(
                task.num_envs, dtype=torch.bool, device=task.device)
            for step in range(int(config["horizon"])):
                with torch.no_grad():
                    action, _, _ = model.act(
                        actor_obs, critic_obs, deterministic=True)
                actor_obs, critic_obs, _, _, terms = env.step(action, step + 1)
                current_count = int((task.successes > 0).sum().item())
                peak_count = max(peak_count, current_count)
                max_height = torch.maximum(max_height, terms["height_delta"])
                ever_failure |= terms["failure_penalty"] < 0
            per_object.append({
                "object_id": object_id,
                "category": object_id.split("-", 2)[1],
                "trajectory_count": task.num_envs,
                "official_peak_success_count": peak_count,
                "official_peak_success_rate": peak_count / task.num_envs,
                "mean_maximum_lift_m": float(max_height.mean().item()),
                "failure_rate": float(ever_failure.float().mean().item()),
            })
        finally:
            if task is not None:
                task.clean_sim()
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "macro_official_peak_success_rate": float(np.mean([
            item["official_peak_success_rate"] for item in per_object])),
        "macro_mean_maximum_lift_m": float(np.mean([
            item["mean_maximum_lift_m"] for item in per_object])),
        "macro_failure_rate": float(np.mean([
            item["failure_rate"] for item in per_object])),
        "total_success_count": int(sum(
            item["official_peak_success_count"] for item in per_object)),
        "total_trajectory_count": int(sum(
            item["trajectory_count"] for item in per_object)),
        "objects": per_object,
    }
    model.train()
    return result


def run(cli):
    absolute_paths(cli)
    config = load_run_config(cli)
    if cli.print_config:
        print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        return

    evaluation_support.initialize_cuda_runtime()
    evaluation_support.require_free_vram(cli.min_free_vram_mb)
    evaluation_support.initialize_runtime()
    import torch
    from custom_tools.residual_env import ResidualDexGraspEnv
    from custom_tools.residual_ppo import (
        PPOConfig, ResidualActorCritic, RolloutStorage, ppo_update)

    trajectory_selection = None
    anchor_env_flags = None
    if cli.trajectory_selection:
        with open(cli.trajectory_selection, "r", encoding="utf-8") as handle:
            trajectory_selection = yaml.safe_load(handle)
        if trajectory_selection.get("status") != "frozen_stage1_selection":
            raise ValueError("Trajectory selection must be frozen before training")
        if Path(trajectory_selection["trajectory_root"]).resolve() != Path(
                cli.trajectory_root).resolve():
            raise ValueError("Training and selection trajectory roots differ")
        object_ids = trajectory_selection["object_ids"]
        trajectory_data = indexed_trajectory_data(
            cli.trajectory_root, object_ids,
            trajectory_selection["trajectory_indices_by_object"])
        if "anchor_flags_by_object" in trajectory_selection:
            anchor_env_flags = []
            for object_id in object_ids:
                flags = trajectory_selection["anchor_flags_by_object"][object_id]
                indices = trajectory_selection[
                    "trajectory_indices_by_object"][object_id]
                if len(flags) != len(indices):
                    raise ValueError("Anchor flags and trajectory indices differ")
                anchor_env_flags.extend(bool(value) for value in flags)
    else:
        object_ids = selected_objects(
            cli.manifest, int(config["objects_per_category"]))
        trajectory_data = sliced_trajectory_data(
            cli.trajectory_root, object_ids, int(config["trajectory_index"]),
            int(config["trajectories_per_object"]))
    total_envs = sum(data["grasp_seqs"].shape[0] for data in trajectory_data)
    validation_config = copy.deepcopy(config.get("validation", {}))
    validation_enabled = bool(validation_config.get("enabled", False))
    if validation_enabled:
        validation_root = Path(validation_config["trajectory_root"])
        if not validation_root.is_absolute():
            validation_root = REPO_ROOT / validation_root
        validation_config["trajectory_root"] = str(validation_root.resolve())
        validation_config["object_ids"] = list(object_ids)
        for object_id in object_ids:
            validation_file = validation_root / "{}.npy".format(object_id)
            if not validation_file.is_file():
                raise FileNotFoundError(validation_file)
    cli.seed = int(config["seed"])
    cli.num_envs = total_envs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cli.run_name or "residual_ppo_smoke_seed{}_{}".format(
        config["seed"], timestamp)
    run_dir = REPO_ROOT / "custom_tools/runs/residual_ppo" / run_name
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    shutil.copy2(cli.config, run_dir / "source_config.yaml")
    if cli.trajectory_selection:
        shutil.copy2(
            cli.trajectory_selection, run_dir / "trajectory_selection.yaml")
    if cli.resume_checkpoint:
        shutil.copy2(cli.resume_checkpoint, run_dir / "resume_checkpoint.pt")

    original_cwd = Path.cwd()
    task = None
    started = time.perf_counter()
    try:
        os.chdir(str(DEXGRASP_ROOT))
        official_args = evaluation_support.build_official_args(cli)
        base_cfg, cfg_train, _ = evaluation_support.load_cfg(official_args)
        evaluation_support.set_seed(
            config["seed"], cfg_train.get("torch_deterministic", False))
        bc_model, _, checkpoint_path, checkpoint = evaluation_support.load_model(cli)
        task = build_task(
            cli, config, official_args, base_cfg, cfg_train, trajectory_data)
        env, actor_obs, critic_obs = create_residual_env(
            task, bc_model, config, ResidualDexGraspEnv)
        category_names, category_masks, advantage_groups = category_context(
            task, object_ids, torch)
        actor_obs_dim = env.actor_obs_dim
        critic_obs_dim = env.critic_obs_dim
        model = ResidualActorCritic(
            actor_obs_dim, critic_obs_dim,
            hidden_dims=tuple(config["hidden_dims"]),
            init_std=config["init_std"],
            gate_dim=2 if config.get("gated_residual", False) else 0,
            initial_gate=config.get("initial_gate", 0.1)).to(cli.rl_device)
        ppo_config = PPOConfig(**config["ppo"])
        anchor_env_mask = None
        if anchor_env_flags is not None:
            anchor_env_mask = torch.tensor(
                anchor_env_flags, dtype=torch.bool, device=cli.rl_device)
        if (ppo_config.anchor_effective_residual_coef > 0
                or ppo_config.anchor_gate_coef > 0) and anchor_env_mask is None:
            raise ValueError(
                "Anchor regularization requires anchor_flags_by_object")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=ppo_config.learning_rate)

        resume = None
        start_iteration = 0
        initial_global_step = 0
        if cli.resume_checkpoint:
            resume = torch.load(
                cli.resume_checkpoint, map_location=cli.rl_device,
                weights_only=False)
            if resume["actor_obs_dim"] != actor_obs_dim:
                raise ValueError("Resume actor observation dimension differs")
            if resume["critic_obs_dim"] != critic_obs_dim:
                raise ValueError("Resume critic observation dimension differs")
            if resume["object_ids"] != object_ids:
                raise ValueError("Resume object order differs")
            model.load_state_dict(resume["model_state_dict"])
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            start_iteration = int(resume["iteration"])
            initial_global_step = int(resume["global_step"])
            if start_iteration >= int(config["iterations"]):
                raise ValueError(
                    "Resume iteration must be smaller than target iterations")

        metadata = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "bc_checkpoint": str(checkpoint_path),
            "bc_checkpoint_epoch": checkpoint.get("epoch"),
            "object_ids": object_ids,
            "num_envs": total_envs,
            "actor_obs_dim": actor_obs_dim,
            "critic_obs_dim": critic_obs_dim,
            "policy_action_dim": model.action_dim,
            "gated_residual": bool(config.get("gated_residual", False)),
            "official_success_definition_changed": False,
            "reward_kind": "custom_residual_ppo_training_reward",
            "trajectory_selection": cli.trajectory_selection or None,
            "resume_checkpoint": cli.resume_checkpoint or None,
            "start_iteration": start_iteration,
            "validation": validation_config if validation_enabled else None,
            "anchor_environment_count": (
                int(anchor_env_mask.sum().item())
                if anchor_env_mask is not None else 0),
        }
        with (run_dir / "metadata.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(metadata, handle, allow_unicode=True, sort_keys=False)
        with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

        torch.cuda.reset_peak_memory_stats()
        fields = None
        metrics_path = run_dir / "metrics.csv"
        validation_metrics_path = run_dir / "validation_metrics.csv"
        global_step = initial_global_step
        completed_iteration = start_iteration
        best_validation = (
            copy.deepcopy(resume.get("validation_metrics"))
            if resume is not None else None)
        latest_validation = copy.deepcopy(best_validation)
        no_improvement_evaluations = 0
        validation_evaluations = 0
        stop_reason = "maximum_iterations"
        with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
            writer = None
            for iteration in range(
                    start_iteration + 1, int(config["iterations"]) + 1):
                storage = RolloutStorage()
                term_sums = {}
                term_squares = {}
                term_absolute_sums = {}
                category_stats = {
                    category: {
                        "reward_sum": 0.0, "reward_square_sum": 0.0,
                        "count": 0, "success_events": 0, "done_events": 0}
                    for category in category_masks}
                success_events = 0
                done_events = 0
                for rollout_step in range(int(config["rollout_steps"])):
                    with torch.no_grad():
                        action, log_prob, value = model.act(actor_obs, critic_obs)
                    next_actor_obs, next_critic_obs, reward, done, terms = env.step(
                        action, global_step + rollout_step + 1)
                    storage.add(
                        actor_obs, critic_obs, action, log_prob, value, reward, done)
                    for name, tensor in terms.items():
                        term_sums[name] = term_sums.get(name, 0.0) + tensor.sum().item()
                        term_squares[name] = (
                            term_squares.get(name, 0.0) + tensor.square().sum().item())
                        term_absolute_sums[name] = (
                            term_absolute_sums.get(name, 0.0) + tensor.abs().sum().item())
                    for category, mask in category_masks.items():
                        stats = category_stats[category]
                        category_reward = reward[mask]
                        stats["reward_sum"] += category_reward.sum().item()
                        stats["reward_square_sum"] += category_reward.square().sum().item()
                        stats["count"] += int(mask.sum().item())
                        stats["success_events"] += int(
                            (terms["success_bonus"][mask] > 0).sum().item())
                        stats["done_events"] += int(done[mask].sum().item())
                    success_events += int((terms["success_bonus"] > 0).sum().item())
                    done_events += int(done.sum().item())
                    if done.any():
                        next_actor_obs, next_critic_obs = env.reset_done(done)
                    actor_obs, critic_obs = next_actor_obs, next_critic_obs
                global_step += int(config["rollout_steps"]) * total_envs
                with torch.no_grad():
                    next_value = model.critic(critic_obs).squeeze(-1)
                batch = storage.finish(
                    next_value, ppo_config, advantage_groups=advantage_groups,
                    anchor_env_mask=anchor_env_mask)
                with torch.no_grad():
                    reconstructed_log_prob, _, _ = model.evaluate(
                        batch["actor_obs"], batch["critic_obs"], batch["actions"])
                    log_prob_reconstruction_error = (
                        reconstructed_log_prob - batch["old_log_probs"]
                    ).abs().max().item()
                    action_saturation_fraction = (
                        batch["actions"].abs() > 0.999).float().mean().item()
                update_metrics = ppo_update(model, optimizer, batch, ppo_config)
                sample_count = int(config["rollout_steps"]) * total_envs
                reward_statistics = reward_term_statistics(
                    term_sums, term_squares, term_absolute_sums, sample_count)
                category_metrics = {}
                for category, stats in category_stats.items():
                    mean = stats["reward_sum"] / stats["count"]
                    variance = max(
                        stats["reward_square_sum"] / stats["count"] - mean * mean,
                        0.0)
                    category_metrics.update({
                        "category_{}_reward_mean".format(category): mean,
                        "category_{}_reward_std".format(category): variance ** 0.5,
                        "category_{}_success_events".format(category):
                            stats["success_events"],
                        "category_{}_done_events".format(category):
                            stats["done_events"],
                    })
                for group_index, stats in batch["advantage_group_stats"].items():
                    category = category_names[group_index]
                    for metric_name, value in stats.items():
                        category_metrics[
                            "category_{}_advantage_{}".format(
                                category, metric_name)] = value
                row = {
                    "iteration": iteration,
                    "global_step": global_step,
                    "success_events": success_events,
                    "done_events": done_events,
                    "policy_std_mean": model.log_std.exp().mean().item(),
                    "wrist_gate_mean": reward_statistics.get(
                        "reward_wrist_gate_mean", 1.0),
                    "finger_gate_mean": reward_statistics.get(
                        "reward_finger_gate_mean", 1.0),
                    "action_saturation_fraction": action_saturation_fraction,
                    "log_prob_reconstruction_error": log_prob_reconstruction_error,
                    "elapsed_seconds": time.perf_counter() - started,
                    **reward_statistics,
                    **category_metrics,
                    **update_metrics,
                }
                if writer is None:
                    fields = list(row.keys())
                    writer = csv.DictWriter(metrics_file, fieldnames=fields)
                    writer.writeheader()
                writer.writerow(row)
                metrics_file.flush()
                print(
                    "iteration={}/{} reward={:.4f} successes={} "
                    "policy_loss={:.4f} value_loss={:.4f}".format(
                        iteration, config["iterations"], row["reward_reward_mean"],
                        success_events, row["policy_loss"], row["value_loss"]))

                completed_iteration = iteration
                should_validate = (
                    validation_enabled
                    and iteration % int(validation_config["interval"]) == 0)
                checkpoint_interval = int(config.get("checkpoint_interval", 0))
                should_checkpoint = (
                    checkpoint_interval > 0
                    and iteration % checkpoint_interval == 0)
                if should_checkpoint and not should_validate:
                    checkpoints_dir = run_dir / "checkpoints"
                    checkpoints_dir.mkdir(exist_ok=True)
                    save_checkpoint(
                        checkpoints_dir / "iteration_{:04d}.pt".format(iteration),
                        model, optimizer, config, iteration, global_step,
                        actor_obs_dim, critic_obs_dim, object_ids,
                        validation_metrics=latest_validation)
                if not should_validate:
                    continue

                # Save completed training before creating validation simulators.
                # If Isaac Gym fails while repeatedly rebuilding simulations,
                # the latest optimization work remains recoverable.
                checkpoints_dir = run_dir / "checkpoints"
                checkpoints_dir.mkdir(exist_ok=True)
                save_checkpoint(
                    checkpoints_dir / "iteration_{:04d}.pt".format(iteration),
                    model, optimizer, config, iteration, global_step,
                    actor_obs_dim, critic_obs_dim, object_ids,
                    validation_metrics=latest_validation)

                # Keep only one Isaac Gym simulation alive at a time.  Model
                # and optimizer state remain in memory; rollout state is
                # intentionally reset after validation at an iteration boundary.
                task.clean_sim()
                task = None
                del env
                gc.collect()
                torch.cuda.empty_cache()
                validation_metrics = evaluate_validation_policy(
                    cli, config, validation_config, model, bc_model,
                    official_args, base_cfg, cfg_train,
                    ResidualDexGraspEnv, torch)
                validation_evaluations += 1
                latest_validation = validation_metrics
                validation_row = {
                    "iteration": iteration,
                    "global_step": global_step,
                    "macro_official_peak_success_rate": validation_metrics[
                        "macro_official_peak_success_rate"],
                    "macro_mean_maximum_lift_m": validation_metrics[
                        "macro_mean_maximum_lift_m"],
                    "macro_failure_rate": validation_metrics[
                        "macro_failure_rate"],
                    "total_success_count": validation_metrics[
                        "total_success_count"],
                    "total_trajectory_count": validation_metrics[
                        "total_trajectory_count"],
                }
                for item in validation_metrics["objects"]:
                    prefix = "{}_{}".format(item["category"], item["object_id"])
                    validation_row[
                        prefix + "_official_peak_success_rate"] = item[
                            "official_peak_success_rate"]
                    validation_row[prefix + "_mean_maximum_lift_m"] = item[
                        "mean_maximum_lift_m"]
                    validation_row[prefix + "_failure_rate"] = item["failure_rate"]
                write_header = not validation_metrics_path.exists()
                with validation_metrics_path.open(
                        "a", encoding="utf-8", newline="") as validation_file:
                    validation_writer = csv.DictWriter(
                        validation_file, fieldnames=list(validation_row.keys()))
                    if write_header:
                        validation_writer.writeheader()
                    validation_writer.writerow(validation_row)

                checkpoints_dir = run_dir / "checkpoints"
                checkpoints_dir.mkdir(exist_ok=True)
                save_checkpoint(
                    checkpoints_dir / "iteration_{:04d}.pt".format(iteration),
                    model, optimizer, config, iteration, global_step,
                    actor_obs_dim, critic_obs_dim, object_ids,
                    validation_metrics=validation_metrics)
                if validation_is_better(
                        validation_metrics, best_validation,
                        lift_tolerance_m=float(
                            validation_config.get("lift_tolerance_m", 1e-3))):
                    best_validation = copy.deepcopy(validation_metrics)
                    no_improvement_evaluations = 0
                    save_checkpoint(
                        run_dir / "best.pt", model, optimizer, config,
                        iteration, global_step, actor_obs_dim, critic_obs_dim,
                        object_ids, validation_metrics=validation_metrics)
                    best_label = "new_best"
                else:
                    no_improvement_evaluations += 1
                    best_label = "no_improvement_{}/{}".format(
                        no_improvement_evaluations,
                        validation_config["patience"])
                print(
                    "validation iteration={} macro_success={:.2f}% "
                    "macro_lift={:.3f}m failure={:.2f}% {}".format(
                        iteration,
                        100.0 * validation_metrics[
                            "macro_official_peak_success_rate"],
                        validation_metrics["macro_mean_maximum_lift_m"],
                        100.0 * validation_metrics["macro_failure_rate"],
                        best_label))

                if no_improvement_evaluations >= int(
                        validation_config["patience"]):
                    stop_reason = "validation_patience_exhausted"
                    break
                if iteration >= int(config["iterations"]):
                    break

                task = build_task(
                    cli, config, official_args, base_cfg, cfg_train,
                    trajectory_data)
                env, actor_obs, critic_obs = create_residual_env(
                    task, bc_model, config, ResidualDexGraspEnv)
                if (env.actor_obs_dim != actor_obs_dim
                        or env.critic_obs_dim != critic_obs_dim):
                    raise RuntimeError("Rebuilt training observation dimensions changed")
                category_names, category_masks, advantage_groups = category_context(
                    task, object_ids, torch)

        save_checkpoint(
            run_dir / "last.pt", model, optimizer, config,
            completed_iteration, global_step,
            actor_obs_dim, critic_obs_dim, object_ids,
            validation_metrics=latest_validation)
        resource = {
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_allocated_mib": float(
                torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "peak_reserved_mib": float(
                torch.cuda.max_memory_reserved() / (1024 ** 2)),
            "completed_iteration": completed_iteration,
            "stop_reason": stop_reason,
            "validation_evaluations": validation_evaluations,
            "best_validation": best_validation,
        }
        with (run_dir / "resource_summary.yaml").open(
                "w", encoding="utf-8") as handle:
            yaml.safe_dump(resource, handle, sort_keys=False)
        print("Saved residual PPO smoke run: {}".format(run_dir))
        print("Resource summary: {}".format(resource))
        return run_dir
    finally:
        if task is not None:
            task.clean_sim()
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    run(parse_cli())
