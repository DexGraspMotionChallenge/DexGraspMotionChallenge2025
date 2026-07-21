"""Convert raw GraspM3 trajectories into BC features outside official code.

The default ``official_final`` selection reproduces the original preprocessor:
only trajectories marked successful at the final replay frame are retained.
Alternative selections are explicit diagnostics and are stored in output
metadata, so they cannot be confused with the challenge baseline.
"""

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for import_root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

SELECTIONS = ("official_final", "ever_task_success", "lift_30cm", "all")


def initialize_cuda_runtime():
    """Load only enough CUDA state to enforce the VRAM safety gate."""
    global torch
    import isaacgym  # noqa: F401
    import torch as torch_module
    torch = torch_module


def initialize_runtime():
    """Import full simulation modules only when conversion will run."""
    global np, unscale
    global get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
    global parse_task, get_AgentIndex

    import numpy as numpy_module
    from isaacgym.torch_utils import unscale as gym_unscale
    from utils.config import (
        get_args as official_get_args,
        load_cfg as official_load_cfg,
        parse_sim_params as official_parse_sim_params,
        set_np_formatting as official_set_np_formatting,
        set_seed as official_set_seed,
    )
    from utils.parse_task import parse_task as official_parse_task
    from utils.process_marl import get_AgentIndex as official_agent_index

    np = numpy_module
    unscale = gym_unscale
    get_args = official_get_args
    load_cfg = official_load_cfg
    parse_sim_params = official_parse_sim_params
    set_np_formatting = official_set_np_formatting
    set_seed = official_set_seed
    parse_task = official_parse_task
    get_AgentIndex = official_agent_index


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Preprocess raw GraspM3 trajectories into a separate output directory.")
    parser.add_argument("--object-id", action="append", default=[])
    parser.add_argument("--manifest", default="")
    parser.add_argument(
        "--manifest-split", action="append", choices=("train", "test", "backups"),
        default=[])
    parser.add_argument("--input-root", default=str(REPO_ROOT / "external_data/dataset"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selection", choices=SELECTIONS, default="official_final")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--min-free-vram-mb", type=int, default=4500,
        help="Abort before preprocessing if less VRAM is free (default: 4500 MiB).")
    return parser.parse_args()


def requested_object_ids(cli):
    object_ids = list(cli.object_id)
    if cli.manifest:
        manifest_path = Path(cli.manifest).expanduser().resolve()
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        split_names = cli.manifest_split or ["train", "test"]
        for category in manifest["criteria"]["categories"]:
            category_data = manifest["categories"][category]
            for split_name in split_names:
                object_ids.extend(category_data[split_name])
    object_ids = list(dict.fromkeys(object_ids))
    if not object_ids:
        raise ValueError("provide --object-id or --manifest")
    return object_ids


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
            "--logdir=logs/dexrep_custom_preprocess",
            "--headless",
        ]
        return get_args()
    finally:
        sys.argv = original_argv


def require_free_vram(min_free_vram_mb):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; preprocessing requires Isaac Gym on GPU.")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_mb = free_bytes / (1024 ** 2)
    total_mb = total_bytes / (1024 ** 2)
    print("GPU memory before preprocessing: {:.0f}/{:.0f} MiB free".format(
        free_mb, total_mb))
    if free_mb < min_free_vram_mb:
        raise RuntimeError(
            "Only {:.0f} MiB VRAM is free, below the safety threshold of {} MiB. "
            "Wait for the other GPU process to finish."
            .format(free_mb, min_free_vram_mb)
        )


def selection_mask(selection, final_success, ever_success, maximum_lift):
    if selection == "official_final":
        return final_success
    if selection == "ever_task_success":
        return ever_success
    if selection == "lift_30cm":
        return maximum_lift >= 0.30
    return torch.ones_like(final_success, dtype=torch.bool)


def process_batch(
        official_args, cfg, cfg_train, sim_params, agent_index, npy_list,
        selection="official_final"):
    """Replay one batch and return one processed dictionary per object."""
    task = None
    env = None
    try:
        task, env = parse_task(
            official_args,
            copy.deepcopy(cfg),
            cfg_train,
            sim_params,
            agent_index,
            npy_list=copy.deepcopy(npy_list),
        )
        sequence = task.grasp_seqs
        num_envs, sequence_length, _ = sequence.shape
        all_obs = torch.zeros(num_envs, sequence_length, 2582)
        gt_all_actions = torch.zeros(num_envs, sequence_length, 28)
        sim_all_actions = torch.zeros(num_envs, sequence_length, 28)
        obj_all_state = torch.zeros(num_envs, sequence_length, 7)

        task.reset_buf = torch.ones(num_envs, device=task.device, dtype=torch.long)
        task.progress_buf = torch.zeros(num_envs, device=task.device, dtype=torch.long)
        env.reset()
        initial_z = task.object_pos[:, 2].clone()
        maximum_z = initial_z.clone()
        ever_success = task.successes > 0

        all_obs[:, 0, :] = task.obs_buf
        obj_all_state[:, 0, :] = task.get_object_state()
        for frame_index in range(1, sequence_length):
            actions = sequence[:, frame_index, :]
            task.step(actions, frame_index)
            gt_all_actions[:, frame_index - 1, :] = actions
            sim_all_actions[:, frame_index - 1, :] = task.shadow_hand_dof_pos
            all_obs[:, frame_index, :] = task.obs_buf
            obj_all_state[:, frame_index, :] = task.get_object_state()
            maximum_z[:] = torch.maximum(maximum_z, task.object_pos[:, 2])
            ever_success |= task.successes > 0

        final_success = task.successes > 0
        maximum_lift = maximum_z - initial_z
        selected = selection_mask(
            selection, final_success, ever_success, maximum_lift).to(all_obs.device)

        outputs = []
        for object_index in np.unique(task.object_idxs):
            object_mask_cpu = task.get_obj_idx_mask(object_index).to(dtype=torch.bool)
            object_mask_gpu = object_mask_cpu.to(final_success.device)
            selected_indices = torch.where(selected & object_mask_cpu)[0].to(all_obs.device)
            gt_actions = gt_all_actions[selected_indices].cpu()
            vis_actions = unscale(
                gt_actions,
                task.shadow_hand_dof_lower_limits.cpu(),
                task.shadow_hand_dof_upper_limits.cpu(),
            )
            output = {
                "obs": all_obs[selected_indices].cpu().numpy(),
                "vis_unscale_actions": torch.clamp(
                    vis_actions, min=-1.0, max=1.0).numpy(),
                "success_idx": selected_indices.cpu().numpy(),
                "selection_metric": selection,
                "official_final_success_idx": torch.where(
                    final_success & object_mask_gpu)[0].cpu().numpy(),
                "ever_task_success_idx": torch.where(
                    ever_success & object_mask_gpu)[0].cpu().numpy(),
                "lift_30cm_idx": torch.where(
                    (maximum_lift >= 0.30) & object_mask_gpu)[0].cpu().numpy(),
                "maximum_lift": maximum_lift[object_mask_gpu].cpu().numpy(),
                "obj_rotmat": task.obj_trajs_info["obj_rotmat"][
                    selected_indices.cpu().numpy()],
                "obj_scale": task.obj_trajs_info["obj_scale"][
                    selected_indices.cpu().numpy()],
                "grasp_seqs": task.obj_trajs_info["grasp_seqs"][
                    selected_indices.cpu().numpy()],
            }
            if len(selected_indices) > 0:
                hand_pcds, _ = task.get_seq_hand_pcd(
                    sim_all_actions.cpu(), selected_indices)
                obj_pcds = task.get_seq_obj_pcd(
                    obj_all_state.cpu(), selected_indices, object_index)
                output["hand_pcds"] = hand_pcds.numpy()
                output["obj_pcds"] = obj_pcds.numpy()
            if "obj_code_idx" in task.obj_trajs_info:
                output["obj_code_idx"] = task.obj_trajs_info["obj_code_idx"][object_index]
            object_code = task.object_code_list[object_index]
            outputs.append((object_code, output))
            print(
                "{}: retained {}/{} with selection={}".format(
                    object_code, len(selected_indices),
                    int(object_mask_cpu.sum().item()), selection))
        return outputs
    finally:
        if task is not None:
            task.clean_sim()
        del env, task
        gc.collect()
        torch.cuda.empty_cache()


def prepare_runtime(seed):
    official_args = build_official_args(seed)
    cfg, cfg_train, _ = load_cfg(official_args)
    cfg["env"].setdefault("seq_start_rot_uniform", False)
    sim_params = parse_sim_params(official_args, cfg, cfg_train)
    set_seed(seed, cfg_train.get("torch_deterministic", False))
    return official_args, cfg, cfg_train, sim_params, get_AgentIndex(cfg)


def data_preprocess_online(data, batch_size=1, selection="official_final", seed=0):
    """In-memory conversion used by our custom dataset loader."""
    initialize_cuda_runtime()
    initialize_runtime()
    if selection not in SELECTIONS:
        raise ValueError("unknown selection: {}".format(selection))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    original_cwd = Path.cwd()
    os.chdir(str(DEXGRASP_ROOT))
    try:
        runtime = prepare_runtime(seed)
        outputs = []
        for start in range(0, len(data), batch_size):
            batch_outputs = process_batch(
                *runtime, npy_list=data[start:start + batch_size], selection=selection)
            outputs.extend(output for _, output in batch_outputs)
        return outputs
    finally:
        os.chdir(str(original_cwd))


def run(cli):
    initialize_cuda_runtime()
    if cli.selection not in SELECTIONS:
        raise ValueError("unknown selection: {}".format(cli.selection))
    if cli.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    require_free_vram(cli.min_free_vram_mb)
    initialize_runtime()

    input_root = Path(cli.input_root).expanduser().resolve()
    output_root = Path(cli.output_root).expanduser().resolve()
    raw_items = []
    for object_id in requested_object_ids(cli):
        input_path = input_root / (object_id + ".npy")
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        output_path = output_root / (object_id + ".npy")
        if output_path.exists() and not cli.overwrite:
            raise FileExistsError(
                "Output already exists: {}. Choose another directory or pass --overwrite."
                .format(output_path))
        item = np.load(str(input_path), allow_pickle=True).item()
        item["obj_code"] = object_id
        raw_items.append(item)

    set_np_formatting()
    original_cwd = Path.cwd()
    os.chdir(str(DEXGRASP_ROOT))
    try:
        runtime = prepare_runtime(cli.seed)
        output_root.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(raw_items), cli.batch_size):
            batch_outputs = process_batch(
                *runtime,
                npy_list=raw_items[start:start + cli.batch_size],
                selection=cli.selection,
            )
            for object_code, output in batch_outputs:
                output_path = output_root / (object_code + ".npy")
                np.save(str(output_path), output, allow_pickle=True)
                print("Saved {}".format(output_path))
    finally:
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    run(parse_cli())
