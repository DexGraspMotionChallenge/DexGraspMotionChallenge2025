"""Collect one DAgger-style round on training trajectories only.

The unified student controls the hand.  A fixed category teacher labels each
student-visited state, but never controls the simulator.  States after an
environment requests reset are excluded.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from custom_tools import evaluate_bc as evaluation_support  # noqa: E402
from custom_tools.diagnose_bc_closed_loop import (  # noqa: E402
    process_observation, reset_task)
from custom_tools.train_residual_ppo import build_task  # noqa: E402


CATEGORIES = ("bottle", "mug", "bowl", "camera")


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Collect student-visited states with routed teacher labels.")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher", action="append", required=True)
    parser.add_argument("--manifest", default=str(
        REPO_ROOT / "custom_tools/configs/object_split_final.json"))
    parser.add_argument("--trajectory-root", default=str(
        DEXGRASP_ROOT / "dataset/bc_multicategory_train"))
    parser.add_argument("--bc-config", default=str(
        REPO_ROOT / "custom_tools/configs/unified_student_distill.yaml"))
    parser.add_argument("--env-config", default=str(
        DEXGRASP_ROOT / "cfg/shadow_hand_grasp_dexrep_ijrr.yaml"))
    parser.add_argument("--train-config", default=str(
        DEXGRASP_ROOT / "cfg/ppo1/config.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=69)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--max-trajectories-per-object", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--num-envs", type=int, default=0)
    return parser.parse_args()


def absolute(path):
    return Path(path).expanduser().resolve()


def parse_teachers(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--teacher must be CATEGORY=CHECKPOINT")
        category, checkpoint = value.split("=", 1)
        checkpoint = absolute(checkpoint)
        if category not in CATEGORIES or category in result:
            raise ValueError("Invalid or duplicate teacher category: {}".format(category))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result[category] = checkpoint
    if set(result) != set(CATEGORIES):
        raise ValueError("Exactly four category teachers are required")
    return result


def trajectory_data(path, object_id, limit):
    source = np.load(path / (object_id + ".npy"), allow_pickle=True).item()
    count = len(source["grasp_seqs"])
    if limit > 0:
        count = min(count, limit)
    data = {"obj_code": object_id}
    for key in ("obj_scale", "obj_rotmat", "grasp_seqs"):
        data[key] = source[key][:count].copy()
    return data, count


def load_bc(cli, checkpoint):
    cli.bc_checkpoint = str(checkpoint)
    model, model_name, path, _ = evaluation_support.load_model(cli)
    if model_name != "ActorCriticDexRep":
        raise ValueError("Online collection requires ActorCriticDexRep")
    return model.eval(), path


def main():
    cli = parse_cli()
    if cli.horizon < 1 or cli.horizon > 122:
        raise ValueError("--horizon must be in [1, 122]")
    output = absolute(cli.output)
    if output.exists():
        raise FileExistsError(output)
    cli.student_checkpoint = str(absolute(cli.student_checkpoint))
    cli.manifest = str(absolute(cli.manifest))
    cli.trajectory_root = str(absolute(cli.trajectory_root))
    cli.bc_config = str(absolute(cli.bc_config))
    cli.env_config = str(absolute(cli.env_config))
    cli.train_config = str(absolute(cli.train_config))
    teachers = parse_teachers(cli.teacher)

    with open(cli.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    selected = [(category, object_id) for category in CATEGORIES
                for object_id in manifest["categories"][category]["train"]]
    if cli.max_objects > 0:
        selected = selected[:cli.max_objects]
    if not selected:
        raise ValueError("No training objects selected")

    evaluation_support.initialize_cuda_runtime()
    evaluation_support.require_free_vram(cli.min_free_vram_mb)
    evaluation_support.initialize_runtime()
    import torch

    original_cwd = Path.cwd()
    task = None
    observations = []
    teacher_actions = []
    student_actions = []
    category_indices = []
    object_indices = []
    trajectory_indices = []
    frame_indices = []
    object_summaries = []
    student = None
    teacher = None
    try:
        os.chdir(str(DEXGRASP_ROOT))
        official_args = evaluation_support.build_official_args(cli)
        base_cfg, cfg_train, _ = evaluation_support.load_cfg(official_args)
        evaluation_support.set_seed(
            cli.seed, cfg_train.get("torch_deterministic", False))
        student, student_path = load_bc(cli, cli.student_checkpoint)
        pro_dim = 100
        current_category = None
        for object_index, (category, object_id) in enumerate(selected):
            if category != current_category:
                if teacher is not None:
                    del teacher
                    gc.collect()
                    torch.cuda.empty_cache()
                teacher, _ = load_bc(cli, teachers[category])
                current_category = category
            data, count = trajectory_data(
                Path(cli.trajectory_root), object_id,
                cli.max_trajectories_per_object)
            cli.num_envs = count
            task = build_task(
                cli, {"seed": cli.seed}, official_args, base_cfg, cfg_train, [data])
            raw_obs = reset_task(task, torch)
            active = torch.ones(count, dtype=torch.bool, device=task.device)
            collected_before = len(category_indices)
            disagreement_sum = 0.0
            disagreement_count = 0
            peak_success = 0
            for frame in range(cli.horizon):
                processed = process_observation(raw_obs, pro_dim, torch)
                with torch.no_grad():
                    student_action = student.model.act_inference(processed).clamp(-1.0, 1.0)
                    teacher_action = teacher.model.act_inference(processed).clamp(-1.0, 1.0)
                live = active.nonzero(as_tuple=False).flatten()
                if live.numel() > 0:
                    observations.append(processed[live].cpu().numpy().astype(np.float32))
                    teacher_actions.append(teacher_action[live].cpu().numpy().astype(np.float32))
                    student_actions.append(student_action[live].cpu().numpy().astype(np.float32))
                    n_live = int(live.numel())
                    category_indices.extend([CATEGORIES.index(category)] * n_live)
                    object_indices.extend([object_index] * n_live)
                    trajectory_indices.extend(live.cpu().tolist())
                    frame_indices.extend([frame] * n_live)
                    disagreement_sum += float(
                        (student_action[live] - teacher_action[live]).abs().sum().item())
                    disagreement_count += n_live * 28
                task.step(student_action, frame + 1)
                peak_success = max(
                    peak_success, int((task.successes > 0).sum().item()))
                active &= task.reset_buf <= 0
                raw_obs = task.obs_buf.clone()
            object_summaries.append({
                "object_id": object_id,
                "category": category,
                "trajectory_count": count,
                "collected_samples": len(category_indices) - collected_before,
                "student_peak_success_count_within_collection_horizon": peak_success,
                "mean_student_teacher_action_mae": (
                    disagreement_sum / disagreement_count
                    if disagreement_count else None),
            })
            task.clean_sim()
            task = None
            gc.collect()
            torch.cuda.empty_cache()
            print("COLLECTED {} samples={} active_end={}".format(
                object_id, object_summaries[-1]["collected_samples"],
                int(active.sum().item())), flush=True)

        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            observations=np.concatenate(observations, axis=0),
            teacher_actions=np.concatenate(teacher_actions, axis=0),
            student_actions=np.concatenate(student_actions, axis=0),
            category_indices=np.asarray(category_indices, dtype=np.int8),
            object_indices=np.asarray(object_indices, dtype=np.int16),
            trajectory_indices=np.asarray(trajectory_indices, dtype=np.int16),
            frame_indices=np.asarray(frame_indices, dtype=np.int16),
            object_ids=np.asarray([item[1] for item in selected]),
        )
        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "method": "one-round pure-student DAgger collection",
            "training_split_only": True,
            "official_code_modified": False,
            "student_checkpoint": str(student_path),
            "teacher_checkpoints": {key: str(value) for key, value in teachers.items()},
            "horizon": cli.horizon,
            "success_count_note": (
                "Diagnostic within the collection horizon; not comparable to "
                "the formal 122-step official evaluation"),
            "sample_count": len(category_indices),
            "objects": object_summaries,
        }
        with output.with_suffix(".yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(summary, handle, allow_unicode=True, sort_keys=False)
        print("ONLINE_IMITATION_COLLECTION=COMPLETE samples={}".format(
            len(category_indices)), flush=True)
    finally:
        if task is not None:
            task.clean_sim()
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    main()
