"""Evaluate many BC checkpoints in one multi-object Isaac Gym simulation."""

import argparse
import gc
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

from custom_tools import evaluate_bc as evaluation_support  # noqa: E402
from custom_tools.train_residual_ppo import build_task, create_residual_env  # noqa: E402


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--bc-config", required=True)
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--object-selection", required=True)
    parser.add_argument(
        "--object-id", default="",
        help="Evaluate only this object (used by the isolated-process driver).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--env-config", default=str(
        DEXGRASP_ROOT / "cfg/shadow_hand_grasp_dexrep_ijrr.yaml"))
    parser.add_argument("--train-config", default=str(
        DEXGRASP_ROOT / "cfg/ppo1/config.yaml"))
    parser.add_argument("--show-viewer", action="store_true")
    return parser.parse_args()


def resolve(cli):
    for name in ("bc_config", "residual_config", "trajectory_root",
                 "object_selection", "output", "env_config", "train_config"):
        setattr(cli, name, str(Path(getattr(cli, name)).expanduser().resolve()))
    cli.checkpoint = [str(Path(path).expanduser().resolve())
                      for path in cli.checkpoint]
    missing = [path for path in cli.checkpoint if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: {}".format(missing))


def evaluate_one(cli, config, task, object_ids, checkpoint_path, torch,
                 residual_env_class, loaded_model=None):
    if loaded_model is None:
        cli.bc_checkpoint = checkpoint_path
        loaded_model = evaluation_support.load_model(cli)
    bc_model, _, resolved_checkpoint, checkpoint = loaded_model
    env, actor_obs, critic_obs = create_residual_env(
        task, bc_model, config, residual_env_class)
    del actor_obs, critic_obs
    object_index = torch.as_tensor(task.object_idxs, device=task.device)
    masks = {index: object_index == index for index in range(len(object_ids))}
    peak_counts = {index: 0 for index in masks}
    peak_masks = {index: torch.zeros(task.num_envs, dtype=torch.bool,
                                    device=task.device) for index in masks}
    ever_failure = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
    maximum_height = torch.full(
        (task.num_envs,), -float("inf"), device=task.device)
    zero = torch.zeros((task.num_envs, env.policy_action_dim), device=task.device)
    for step in range(int(config["horizon"])):
        _, _, _, _, terms = env.step(zero, step + 1)
        success = task.successes > 0
        for index, mask in masks.items():
            count = int((success & mask).sum().item())
            if count > peak_counts[index]:
                peak_counts[index] = count
                peak_masks[index] = success & mask
        ever_failure |= terms["failure_penalty"] < 0
        maximum_height = torch.maximum(maximum_height, terms["height_delta"])
    objects = []
    for index, object_id in enumerate(object_ids):
        mask = masks[index]
        count = int(mask.sum().item())
        objects.append({
            "object_id": object_id,
            "category": object_id.split("-", 2)[1],
            "trajectory_count": count,
            "official_peak_success_count": peak_counts[index],
            "official_peak_success_rate": peak_counts[index] / count,
            "official_peak_success_local_indices": torch.where(
                peak_masks[index][mask])[0].cpu().tolist(),
            "mean_maximum_lift_m": float(maximum_height[mask].mean().item()),
            "failure_rate": float(ever_failure[mask].float().mean().item()),
        })
    result = {
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_epoch": (
            int(checkpoint["epoch"]) if checkpoint.get("epoch") is not None
            else None),
        "total_success_count": sum(x["official_peak_success_count"] for x in objects),
        "total_trajectory_count": sum(x["trajectory_count"] for x in objects),
        "macro_official_peak_success_rate": float(np.mean([
            x["official_peak_success_rate"] for x in objects])),
        "macro_mean_maximum_lift_m": float(np.mean([
            x["mean_maximum_lift_m"] for x in objects])),
        "macro_failure_rate": float(np.mean([x["failure_rate"] for x in objects])),
        "objects": objects,
    }
    result["overall_official_peak_success_rate"] = (
        result["total_success_count"] / result["total_trajectory_count"])
    del env, bc_model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    cli = parse_cli()
    resolve(cli)
    output = Path(cli.output)
    if output.exists():
        raise FileExistsError(output)
    with Path(cli.object_selection).open(encoding="utf-8") as handle:
        selection = yaml.safe_load(handle)
    with Path(cli.residual_config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["seed"] = int(cli.seed)
    object_ids = list(selection["object_ids"])
    if cli.object_id:
        if cli.object_id not in object_ids:
            raise ValueError("Object is absent from selection: {}".format(
                cli.object_id))
        object_ids = [cli.object_id]
    trajectory_data = []
    for object_id in object_ids:
        source = np.load(
            Path(cli.trajectory_root) / (object_id + ".npy"),
            allow_pickle=True).item()
        item = {"obj_code": object_id}
        for key in ("obj_scale", "obj_rotmat", "grasp_seqs"):
            item[key] = source[key].copy()
        trajectory_data.append(item)
    cli.seed = int(cli.seed)
    cli.num_envs = sum(item["grasp_seqs"].shape[0] for item in trajectory_data)
    evaluation_support.initialize_cuda_runtime()
    evaluation_support.require_free_vram(cli.min_free_vram_mb)
    evaluation_support.initialize_runtime()
    import torch
    from custom_tools.residual_env import ResidualDexGraspEnv
    original_cwd = Path.cwd()
    task = None
    first_loaded_model = None
    try:
        import os
        os.chdir(str(DEXGRASP_ROOT))
        official_args = evaluation_support.build_official_args(cli)
        base_cfg, cfg_train, _ = evaluation_support.load_cfg(official_args)
        evaluation_support.set_seed(
            cli.seed, cfg_train.get("torch_deterministic", False))
        # Preserve the verified evaluator's random-number consumption order:
        # load/initialize the BC network before creating the Isaac Gym task.
        cli.bc_checkpoint = cli.checkpoint[0]
        first_loaded_model = evaluation_support.load_model(cli)
        task = build_task(
            cli, config, official_args, base_cfg, cfg_train, trajectory_data)
        results = []
        for index, checkpoint in enumerate(cli.checkpoint, 1):
            print("checkpoint {}/{}: {}".format(
                index, len(cli.checkpoint), checkpoint), flush=True)
            results.append(evaluate_one(
                cli, config, task, object_ids, checkpoint, torch,
                ResidualDexGraspEnv,
                loaded_model=first_loaded_model if index == 1 else None))
            if index == 1:
                first_loaded_model = None
        aggregate = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "evaluation_mode": "single_sim_multi_object_multi_checkpoint",
            "success_metric": "official_peak_per_object",
            "official_success_definition_changed": False,
            "seed": cli.seed,
            "trajectory_root": cli.trajectory_root,
            "object_ids": object_ids,
            "checkpoint_results": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(aggregate, handle, allow_unicode=True, sort_keys=False)
        print("BATCHED_BC_EVALUATION=COMPLETE")
    finally:
        if task is not None:
            task.clean_sim()
        import os
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    main()
