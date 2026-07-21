"""Diagnose raw GraspM3 trajectory replay without modifying official code.

This script creates the official ShadowHandGraspDexRepIjrr2 preprocessing task,
replays one raw object's trajectories, and writes per-trajectory diagnostics.
It never overwrites the input dataset or any preprocessed training file.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
from datetime import datetime

import isaacgym  # Must be imported before torch.
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for import_root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from utils.config import (  # noqa: E402
    get_args,
    load_cfg,
    parse_sim_params,
    set_np_formatting,
    set_seed,
)
from utils.parse_task import parse_task  # noqa: E402
from utils.process_marl import get_AgentIndex  # noqa: E402


def parse_bool(value):
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Replay raw GraspM3 trajectories and report why they fail.")
    parser.add_argument("--object-id", required=True)
    parser.add_argument(
        "--data-root", default=str(REPO_ROOT / "external_data" / "dataset"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-time", type=parse_bool, default=True)
    parser.add_argument("--extra-hold-steps", type=int, default=0)
    parser.add_argument("--control-frequency-inv", type=int, default=3)
    parser.add_argument("--trajectory-indices", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def build_official_args(seed):
    original_argv = sys.argv
    try:
        sys.argv = [
            original_argv[0],
            "--task=ShadowHandGraspDexRepIjrr2",
            "--algo=ppo1",
            "--seed={}".format(seed),
            "--rl_device=cuda:0",
            "--sim_device=cuda:0",
            "--logdir=logs/dexrep_dexgrasp",
            "--headless",
        ]
        return get_args()
    finally:
        sys.argv = original_argv


def select_trajectories(data, index_text):
    trajectory_count = len(data["grasp_seqs"])
    if index_text:
        indices = [int(value.strip()) for value in index_text.split(",") if value.strip()]
    else:
        indices = list(range(trajectory_count))
    invalid = [index for index in indices if index < 0 or index >= trajectory_count]
    if invalid:
        raise IndexError(
            "trajectory indices {} outside [0, {})".format(invalid, trajectory_count))

    selected = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape[0] == trajectory_count:
            selected[key] = value[indices].copy()
        else:
            selected[key] = copy.deepcopy(value)
    return selected, indices


def tensor_list(tensor):
    return tensor.detach().cpu().numpy().tolist()


def run(cli):
    if cli.extra_hold_steps < 0:
        raise ValueError("--extra-hold-steps must be non-negative")
    if cli.control_frequency_inv < 1:
        raise ValueError("--control-frequency-inv must be positive")

    raw_path = Path(cli.data_root).expanduser().resolve() / (cli.object_id + ".npy")
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    raw_data = np.load(str(raw_path), allow_pickle=True).item()
    required = {"obj_rotmat", "obj_scale", "grasp_seqs"}
    missing = required - set(raw_data)
    if missing:
        raise KeyError("raw data missing keys: {}".format(sorted(missing)))
    selected_data, source_indices = select_trajectories(raw_data, cli.trajectory_indices)
    selected_data["obj_code"] = cli.object_id

    original_cwd = Path.cwd()
    os.chdir(str(DEXGRASP_ROOT))
    task = None
    env = None
    try:
        official_args = build_official_args(cli.seed)
        cfg, cfg_train, _ = load_cfg(official_args)
        cfg["env"]["random_time"] = cli.random_time
        cfg["env"]["controlFrequencyInv"] = cli.control_frequency_inv
        # Ijrr2 does not use this key, but keep the in-memory config complete.
        cfg["env"].setdefault("seq_start_rot_uniform", False)
        sim_params = parse_sim_params(official_args, cfg, cfg_train)
        set_seed(cli.seed, cfg_train.get("torch_deterministic", False))
        agent_index = get_AgentIndex(cfg)

        task, env = parse_task(
            official_args,
            cfg,
            cfg_train,
            sim_params,
            agent_index,
            npy_list=[copy.deepcopy(selected_data)],
        )

        sequence = task.grasp_seqs
        num_envs, sequence_length, action_dim = sequence.shape
        task.reset_buf = torch.ones(num_envs, device=task.device, dtype=torch.long)
        task.progress_buf = torch.zeros(num_envs, device=task.device, dtype=torch.long)
        env.reset()

        initial_pos = task.object_pos.clone()
        goal_pos = task.goal_pos.clone()
        max_z = initial_pos[:, 2].clone()
        min_z = initial_pos[:, 2].clone()
        final_pos = initial_pos.clone()
        min_goal_distance = torch.norm(goal_pos - initial_pos, dim=-1)
        max_abs_xy = torch.max(torch.abs(initial_pos[:, :2]), dim=-1).values
        ever_task_success = task.successes > 0
        first_success_step = torch.full(
            (num_envs,), -1, dtype=torch.long, device=task.device)
        tracking_error_sum = torch.zeros(num_envs, device=task.device)
        tracking_error_max = torch.zeros(num_envs, device=task.device)
        tracked_steps = 0
        reset_seen = task.reset_buf > 0

        def record_state(step_index, target_action):
            nonlocal final_pos, min_goal_distance, max_abs_xy, ever_task_success
            nonlocal tracking_error_sum, tracking_error_max, tracked_steps, reset_seen

            final_pos = task.object_pos.clone()
            max_z[:] = torch.maximum(max_z, final_pos[:, 2])
            min_z[:] = torch.minimum(min_z, final_pos[:, 2])
            min_goal_distance[:] = torch.minimum(
                min_goal_distance, torch.norm(goal_pos - final_pos, dim=-1))
            max_abs_xy[:] = torch.maximum(
                max_abs_xy, torch.max(torch.abs(final_pos[:, :2]), dim=-1).values)

            current_success = task.successes > 0
            newly_successful = current_success & ~ever_task_success
            first_success_step[newly_successful] = step_index
            ever_task_success |= current_success
            reset_seen |= task.reset_buf > 0

            error = torch.mean(
                torch.abs(task.shadow_hand_dof_pos - target_action), dim=-1)
            tracking_error_sum += error
            tracking_error_max[:] = torch.maximum(tracking_error_max, error)
            tracked_steps += 1

        for frame_index in range(1, sequence_length):
            target_action = sequence[:, frame_index, :]
            task.step(target_action, frame_index)
            record_state(frame_index, target_action)

        final_action = sequence[:, -1, :]
        for hold_index in range(cli.extra_hold_steps):
            task.step(final_action, sequence_length + hold_index)
            record_state(sequence_length + hold_index, final_action)

        final_success = task.successes > 0
        max_lift = max_z - initial_pos[:, 2]
        final_lift = final_pos[:, 2] - initial_pos[:, 2]
        lift_30cm = max_lift >= 0.30
        mean_tracking_error = tracking_error_sum / max(1, tracked_steps)

        records = []
        for local_index, source_index in enumerate(source_indices):
            records.append({
                "source_index": source_index,
                "initial_position": tensor_list(initial_pos[local_index]),
                "goal_position": tensor_list(goal_pos[local_index]),
                "final_position": tensor_list(final_pos[local_index]),
                "minimum_z": float(min_z[local_index].item()),
                "maximum_z": float(max_z[local_index].item()),
                "maximum_lift": float(max_lift[local_index].item()),
                "final_lift": float(final_lift[local_index].item()),
                "minimum_goal_distance": float(min_goal_distance[local_index].item()),
                "maximum_abs_xy": float(max_abs_xy[local_index].item()),
                "official_final_success": bool(final_success[local_index].item()),
                "task_success_ever": bool(ever_task_success[local_index].item()),
                "lifted_30cm_ever": bool(lift_30cm[local_index].item()),
                "first_success_step": int(first_success_step[local_index].item()),
                "mean_hand_tracking_error": float(mean_tracking_error[local_index].item()),
                "max_hand_tracking_error": float(tracking_error_max[local_index].item()),
                "reset_seen_after_initialization": bool(reset_seen[local_index].item()),
            })

        summary = {
            "object_id": cli.object_id,
            "raw_path": str(raw_path),
            "seed": cli.seed,
            "random_time": cli.random_time,
            "extra_hold_steps": cli.extra_hold_steps,
            "control_frequency_inv": cli.control_frequency_inv,
            "trajectory_count": num_envs,
            "sequence_length": sequence_length,
            "action_dim": action_dim,
            "official_final_success_count": int(final_success.sum().item()),
            "task_success_ever_count": int(ever_task_success.sum().item()),
            "lifted_30cm_count": int(lift_30cm.sum().item()),
            "reset_seen_count": int(reset_seen.sum().item()),
            "mean_maximum_lift": float(max_lift.mean().item()),
            "mean_hand_tracking_error": float(mean_tracking_error.mean().item()),
        }
        result = {"summary": summary, "trajectories": records}
    finally:
        if task is not None:
            task.clean_sim()
        del env, task
        torch.cuda.empty_cache()
        os.chdir(str(original_cwd))

    if cli.output:
        output_path = Path(cli.output).expanduser().resolve()
    else:
        output_dir = REPO_ROOT / "custom_tools" / "results"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / (
            "replay_{}_seed{}_random{}_hold{}_{}.json".format(
                cli.object_id,
                cli.seed,
                int(cli.random_time),
                cli.extra_hold_steps,
                timestamp,
            ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False)

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print("diagnostic_result={}".format(output_path))


if __name__ == "__main__":
    set_np_formatting()
    run(parse_cli())
