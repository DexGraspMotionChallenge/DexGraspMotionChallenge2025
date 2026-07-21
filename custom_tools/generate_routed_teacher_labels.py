"""Generate offline action labels from a fixed category-routed teacher pool."""

import argparse
import gc
import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if root not in sys.path:
        sys.path.insert(0, root)

import isaacgym  # noqa: E402,F401
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel  # noqa: E402
from custom_tools.graspm3_dexrep_dataset import (  # noqa: E402
    GraspM3DexRepDataset)


CATEGORIES = ("bottle", "mug", "bowl", "camera")


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", action="append", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-config", default=str(
        DEXGRASP_ROOT / "cfg/shadow_hand_grasp_dexrep_ijrr.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    return parser.parse_args()


def absolute(path):
    return Path(path).expanduser().resolve()


def parse_teachers(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--teacher must be CATEGORY=CHECKPOINT")
        category, checkpoint = value.split("=", 1)
        if category not in CATEGORIES or category in result:
            raise ValueError("Invalid or duplicate category: {}".format(category))
        checkpoint = absolute(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result[category] = checkpoint
    if set(result) != set(CATEGORIES):
        raise ValueError("Exactly four teachers are required")
    return result


def main():
    cli = parse_cli()
    # Resolve paths before temporarily changing into dexgrasp/.  The dataset
    # expects that working directory, while CLI paths are repository-relative.
    output = absolute(cli.output)
    config_path = absolute(cli.config)
    env_config_path = absolute(cli.env_config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes / (1024 ** 2) < cli.min_free_vram_mb:
        raise RuntimeError("Insufficient free VRAM for teacher labeling")
    teacher_paths = parse_teachers(cli.teacher)
    args = OmegaConf.load(str(config_path))
    args.add_noise = False
    if args.get("distillation") is not None:
        args.distillation.enabled = False
    env_args = OmegaConf.load(str(env_config_path))
    env_args.env.obs_dim.pop("pnG")

    original_cwd = Path.cwd()
    try:
        os.chdir(str(DEXGRASP_ROOT))
        dataset = GraspM3DexRepDataset(args, ds_name="train")
        loader = DataLoader(
            dataset, batch_size=cli.batch_size, shuffle=False,
            num_workers=0, pin_memory=True)
        labels = np.empty((len(dataset), 28), dtype=np.float32)
        filled = np.zeros(len(dataset), dtype=np.bool_)
        for category in CATEGORIES:
            checkpoint = torch.load(
                teacher_paths[category], map_location="cpu")
            teacher = LitBCModel(args, env_args.env)
            teacher.load_state_dict(
                checkpoint.get("state_dict", checkpoint), strict=True)
            teacher = teacher.cuda().eval()
            with torch.no_grad():
                for batch in loader:
                    names = [
                        dataset.obj_code_name_list[int(index)]
                        for index in batch["obj_code_idx"].tolist()]
                    mask = torch.tensor(
                        [name.split("-", 2)[1] == category for name in names],
                        dtype=torch.bool)
                    if not mask.any():
                        continue
                    observations = batch["obs"][mask].cuda(non_blocking=True)
                    actions = teacher.model.act_inference(observations)
                    sample_indices = batch["sample_index"][mask].numpy()
                    labels[sample_indices] = actions.cpu().numpy()
                    filled[sample_indices] = True
            del teacher, checkpoint
            gc.collect()
            torch.cuda.empty_cache()
        if not filled.all():
            raise RuntimeError(
                "Missing teacher labels for {} samples".format((~filled).sum()))
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            teacher_actions=labels,
            sample_count=np.asarray([len(dataset)], dtype=np.int64),
            teacher_categories=np.asarray(CATEGORIES),
            teacher_checkpoints=np.asarray(
                [str(teacher_paths[category]) for category in CATEGORIES]),
        )
        print("ROUTED_TEACHER_LABELS=COMPLETE samples={}".format(
            len(dataset)), flush=True)
    finally:
        os.chdir(str(original_cwd))


if __name__ == "__main__":
    main()
