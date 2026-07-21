"""Freeze policy-blind unseen-v2 objects after expert-replay usability checks."""

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=str(
        REPO_ROOT / "custom_tools/configs/unseen_v2_candidates.json"))
    parser.add_argument("--preprocessed-root", default=str(
        REPO_ROOT / "dexgrasp/dataset/unseen_v2_candidates_preprocessed"))
    parser.add_argument("--output-manifest", default=str(
        REPO_ROOT / "custom_tools/configs/unseen_v2_final.json"))
    parser.add_argument("--output-root", default=str(
        REPO_ROOT / "dexgrasp/dataset/unseen_v2_final"))
    parser.add_argument("--minimum-retained", type=int, default=12)
    return parser.parse_args()


def main():
    cli = parse_cli()
    candidates_path = Path(cli.candidates).expanduser().resolve()
    source_root = Path(cli.preprocessed_root).expanduser().resolve()
    output_manifest = Path(cli.output_manifest).expanduser().resolve()
    output_root = Path(cli.output_root).expanduser().resolve()
    if output_manifest.exists():
        raise FileExistsError(output_manifest)
    with candidates_path.open(encoding="utf-8") as handle:
        candidates = json.load(handle)
    final = {
        "status": "frozen_before_any_learned_policy_evaluation",
        "selection_rule": candidates["selection_rule"],
        "source_candidates": str(candidates_path),
        "expert_replay_filter_only": True,
        "minimum_retained_trajectories": cli.minimum_retained,
        "criteria": dict(candidates["criteria"]),
        "categories": {},
        "objects": {},
    }
    final["criteria"]["backups_per_category"] = 0
    output_root.mkdir(parents=True, exist_ok=True)
    for category in candidates["criteria"]["categories"]:
        selected = list(candidates["categories"][category]["test"])
        final["categories"][category] = {
            "train": [], "test": selected, "backups": []}
        for object_id in selected:
            source = source_root / (object_id + ".npy")
            if not source.is_file():
                raise FileNotFoundError(source)
            data = np.load(str(source), allow_pickle=True).item()
            retained = int(data["grasp_seqs"].shape[0])
            if retained < cli.minimum_retained:
                raise RuntimeError(
                    "{} retains only {} trajectories".format(object_id, retained))
            target = output_root / source.name
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve() != source:
                    raise FileExistsError(target)
            else:
                target.symlink_to(source)
            details = dict(candidates["objects"][object_id])
            details["split"] = "test"
            details["retained_expert_trajectories"] = retained
            final["objects"][object_id] = details
            print("frozen {}: {} trajectories".format(object_id, retained))
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(final, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("UNSEEN_V2=FROZEN_BEFORE_POLICY_EVALUATION")


if __name__ == "__main__":
    main()
