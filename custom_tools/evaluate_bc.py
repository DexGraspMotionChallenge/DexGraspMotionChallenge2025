"""Evaluate a BC checkpoint with custom extensions kept outside official code.

The default success metric reproduces the official evaluator: the largest
number of simultaneously successful environments seen during the rollout.
The ever-success diagnostic and dense diagnostic reward are labeled
separately and never replace the official metric silently.
"""

import argparse
import copy
import gc
import hashlib
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for import_root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import yaml  # noqa: E402


def initialize_cuda_runtime():
    """Load only enough CUDA state to enforce the VRAM safety gate."""
    global torch
    import isaacgym  # noqa: F401
    import torch as torch_module
    torch = torch_module


def initialize_runtime():
    """Import full simulation modules only after the VRAM check passes."""
    global OmegaConf, LitBCModel, test_env
    global CustomShadowHandGraspDexRepIjrr, VecTaskPython
    global get_args, load_cfg, parse_sim_params, set_seed

    from omegaconf import OmegaConf as omega_conf
    from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel as bc_model
    from custom_tools.evaluation_loop import test_env as evaluation_loop
    from custom_tools.shadow_hand_grasp_dexrep_custom import (
        CustomShadowHandGraspDexRepIjrr as custom_task,
    )
    from tasks.hand_base.vec_task import VecTaskPython as vec_task
    from utils.config import (
        get_args as official_get_args,
        load_cfg as official_load_cfg,
        parse_sim_params as official_parse_sim_params,
        set_seed as official_set_seed,
    )

    OmegaConf = omega_conf
    LitBCModel = bc_model
    test_env = evaluation_loop
    CustomShadowHandGraspDexRepIjrr = custom_task
    VecTaskPython = vec_task
    get_args = official_get_args
    load_cfg = official_load_cfg
    parse_sim_params = official_parse_sim_params
    set_seed = official_set_seed


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Evaluate a DexGrasp BC checkpoint without editing official files.")
    parser.add_argument("--object-id", action="append", default=[],
                        help="Object code; repeat this option for several objects.")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--manifest-split", choices=("train", "test"), default="train")
    parser.add_argument("--bc-checkpoint", "--bc_checkpoint", dest="bc_checkpoint", default="")
    parser.add_argument("--bc-config", default=str(
        REPO_ROOT / "ActionDiffusion/bc/config/lhm_bc.yaml"))
    parser.add_argument("--env-config", default=str(
        DEXGRASP_ROOT / "cfg/shadow_hand_grasp_dexrep_ijrr.yaml"))
    parser.add_argument("--train-config", default=str(
        DEXGRASP_ROOT / "cfg/ppo1/config.yaml"))
    parser.add_argument("--result-tag", "--result_tag", dest="result_tag", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=0)
    parser.add_argument("--trajectory-indices", "--trajectory_indices",
                        dest="trajectory_indices", default="")
    parser.add_argument(
        "--trajectory-root", default="",
        help="Directory containing preprocessed <object-id>.npy files.")
    parser.add_argument("--success-metric", choices=("official_peak", "ever"),
                        default="official_peak")
    parser.add_argument("--dense-diagnostic-reward", action="store_true")
    parser.add_argument("--verbose-steps", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--capture-dir", "--capture_dir", dest="capture_dir", default="")
    parser.add_argument("--capture-env", type=int, default=0)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--capture-stride", type=int, default=1)
    parser.add_argument(
        "--min-free-vram-mb", type=int, default=4500,
        help="Abort before evaluation if less VRAM is free (default: 4500 MiB).")
    return parser.parse_args()


def build_official_args(cli):
    """Let Isaac Gym parse only options supported by the untouched project."""
    official_argv = [
        sys.argv[0],
        "--task=ShadowHandGraspDexRepIjrr",
        "--algo=ppo1",
        "--seed={}".format(cli.seed),
        "--rl_device={}".format(cli.rl_device),
        "--sim_device={}".format(cli.sim_device),
        "--logdir=logs/dexrep_custom_eval",
        "--cfg_train={}".format(Path(cli.train_config).expanduser().resolve()),
        "--cfg_env={}".format(Path(cli.env_config).expanduser().resolve()),
    ]
    if not cli.show_viewer:
        official_argv.append("--headless")
    if cli.num_envs > 0:
        official_argv.append("--num_envs={}".format(cli.num_envs))

    original_argv = sys.argv
    try:
        sys.argv = official_argv
        return get_args()
    finally:
        sys.argv = original_argv


def parse_indices(text):
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def require_free_vram(min_free_vram_mb):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Isaac Gym evaluation requires the GPU.")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_mb = free_bytes / (1024 ** 2)
    total_mb = total_bytes / (1024 ** 2)
    print("GPU memory before evaluation: {:.0f}/{:.0f} MiB free".format(
        free_mb, total_mb))
    if free_mb < min_free_vram_mb:
        raise RuntimeError(
            "Only {:.0f} MiB VRAM is free, below the safety threshold of {} MiB. "
            "Wait for the other GPU process to finish."
            .format(free_mb, min_free_vram_mb)
        )


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(cli):
    bc_config_path = Path(cli.bc_config).expanduser().resolve()
    env_config_path = Path(cli.env_config).expanduser().resolve()
    bc_args = OmegaConf.load(str(bc_config_path))
    env_args = OmegaConf.load(str(env_config_path))
    model_name = bc_args.policy.actor_critic

    if model_name == "ActorCriticPNG":
        env_args.env.obs_dim.pop("dexrep_sensor")
        env_args.env.obs_dim.pop("dexrep_pnl")
    elif model_name == "ActorCriticDexRep":
        env_args.env.obs_dim.pop("pnG")

    if cli.bc_checkpoint:
        checkpoint_path = Path(cli.bc_checkpoint).expanduser().resolve()
    else:
        configured = Path(str(bc_args.policy.checkpoints))
        checkpoint_path = configured if configured.is_absolute() else REPO_ROOT / configured
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    model = LitBCModel(bc_args, env_args.env).to(cli.rl_device)
    checkpoint = torch.load(str(checkpoint_path), map_location=torch.device(cli.rl_device))
    model.load_state_dict(checkpoint["state_dict"])
    return model, model_name, checkpoint_path, checkpoint


def create_env(cli, official_args, base_cfg, cfg_train, object_id, capture_dir):
    cfg = copy.deepcopy(base_cfg)
    if cli.trajectory_root:
        cfg["trajs_path"]["train"] = str(Path(cli.trajectory_root).resolve())
    cfg["seed"] = cli.seed
    cfg["env"]["seed"] = cli.seed
    cfg["env"]["env_mode"] = "bc_env_infer"
    cfg["env"]["observationType"] = "DexRep"
    cfg["env"]["obj_type"] = "one"
    cfg["env"]["object_code_dict"] = [object_id]
    cfg["env"]["trajectory_indices"] = parse_indices(cli.trajectory_indices)
    cfg["env"]["diagnostic_dense_reward"] = cli.dense_diagnostic_reward
    cfg["env"]["evaluation_step_logging"] = cli.verbose_steps
    cfg["env"]["capture_dir"] = str(capture_dir) if capture_dir else ""
    cfg["env"]["capture_env"] = cli.capture_env
    cfg["env"]["capture_width"] = cli.capture_width
    cfg["env"]["capture_height"] = cli.capture_height
    cfg["env"]["capture_stride"] = cli.capture_stride
    cfg["env"].setdefault("seq_start_rot_uniform", False)

    sim_params = parse_sim_params(official_args, cfg, cfg_train)
    task = CustomShadowHandGraspDexRepIjrr(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=official_args.physics_engine,
        device_type=official_args.device,
        device_id=official_args.device_id,
        headless=official_args.headless,
        is_multi_agent=False,
    )
    return task, VecTaskPython(task, cli.rl_device)


def safe_tag(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def resolve_cli_paths(cli, invocation_cwd):
    """Resolve user paths before changing into the project's dexgrasp folder."""
    for attribute in (
            "bc_checkpoint", "bc_config", "env_config", "train_config",
            "output", "capture_dir", "trajectory_root", "manifest"):
        value = getattr(cli, attribute)
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = invocation_cwd / path
        setattr(cli, attribute, str(path.resolve()))


def expand_manifest_objects(cli):
    object_ids = list(cli.object_id)
    if cli.manifest:
        with Path(cli.manifest).open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
        for category in manifest["criteria"]["categories"]:
            object_ids.extend(manifest["categories"][category][cli.manifest_split])
    cli.object_id = list(dict.fromkeys(object_ids))
    if not cli.object_id:
        raise ValueError("Provide --object-id or --manifest")


def run(cli):
    if cli.num_envs < 0:
        raise ValueError("--num-envs cannot be negative")
    if cli.capture_stride < 1:
        raise ValueError("--capture-stride must be positive")

    original_cwd = Path.cwd()
    resolve_cli_paths(cli, original_cwd)
    expand_manifest_objects(cli)

    initialize_cuda_runtime()
    require_free_vram(cli.min_free_vram_mb)
    initialize_runtime()
    os.chdir(str(DEXGRASP_ROOT))
    try:
        official_args = build_official_args(cli)
        base_cfg, cfg_train, _ = load_cfg(official_args)
        set_seed(cli.seed, cfg_train.get("torch_deterministic", False))
        model, model_name, checkpoint_path, checkpoint = load_model(cli)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        evaluation_started = time.perf_counter()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = safe_tag(cli.result_tag) or "bc_eval"
        trajectory_indices = parse_indices(cli.trajectory_indices)
        results = {
            "created_at": timestamp,
            "command": " ".join(sys.argv),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "seed": cli.seed,
            "result_tag": tag,
            "success_metric": cli.success_metric,
            "reward_kind": (
                "custom_dense_diagnostic" if cli.dense_diagnostic_reward
                else "official_zero_reward"),
            "trajectory_indices": trajectory_indices,
            "trajectory_root": cli.trajectory_root,
            "manifest": cli.manifest,
            "manifest_split": cli.manifest_split,
            "objects": [],
        }

        for object_id in cli.object_id:
            task = None
            env = None
            try:
                capture_dir = None
                if cli.capture_dir:
                    capture_root = Path(cli.capture_dir).expanduser().resolve()
                    capture_dir = capture_root / object_id if len(cli.object_id) > 1 else capture_root
                task, env = create_env(
                    cli, official_args, base_cfg, cfg_train, object_id, capture_dir)
                metrics, description = test_env(
                    cli, task, env, model, model_name, object_id)
                metrics["object_id"] = object_id
                metrics["description"] = description
                metrics["source_trajectory_indices"] = [
                    trajectory_indices[index] if trajectory_indices else index
                    for index in metrics["success_env_indices"]
                ]
                results["objects"].append(metrics)
            finally:
                if task is not None:
                    task.clean_sim()
                del env, task
                gc.collect()
                torch.cuda.empty_cache()

        results["total_succ_rates"] = sum(
            item["success_rate"] for item in results["objects"]) / len(results["objects"])
        results["total_mean_rewards"] = sum(
            item["mean_rollout_reward"] for item in results["objects"]) / len(results["objects"])
        torch.cuda.synchronize()
        results["elapsed_seconds"] = float(time.perf_counter() - evaluation_started)
        results["peak_allocated_mib"] = float(
            torch.cuda.max_memory_allocated() / (1024 ** 2))
        results["peak_reserved_mib"] = float(
            torch.cuda.max_memory_reserved() / (1024 ** 2))

        if cli.output:
            output_path = Path(cli.output).expanduser().resolve()
        else:
            output_path = (
                REPO_ROOT / "custom_tools/results/evaluations" /
                "{}_seed{}_{}.yaml".format(tag, cli.seed, timestamp))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            yaml.safe_dump(results, output_file, allow_unicode=True, sort_keys=False)
        print("Saved evaluation: {}".format(output_path))
        return output_path
    finally:
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    run(parse_cli())
