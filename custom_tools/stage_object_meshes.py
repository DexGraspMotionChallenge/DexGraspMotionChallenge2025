"""Safely expose external GraspM3 meshes at the path expected by the task."""

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create non-destructive mesh directory symlinks for selected objects.")
    parser.add_argument("--object-id", action="append", default=[])
    parser.add_argument("--manifest", default="")
    parser.add_argument(
        "--manifest-split", action="append", choices=("train", "test", "backups"),
        default=[])
    parser.add_argument(
        "--source-root", default=str(REPO_ROOT / "external_data" / "meshdata"))
    parser.add_argument(
        "--target-root", default=str(REPO_ROOT / "assets" / "meshdata"))
    return parser.parse_args()


def requested_object_ids(args):
    object_ids = list(args.object_id)
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        split_names = args.manifest_split or ["train", "test"]
        for category in manifest["criteria"]["categories"]:
            category_data = manifest["categories"][category]
            for split_name in split_names:
                object_ids.extend(category_data[split_name])
    object_ids = list(dict.fromkeys(object_ids))
    if not object_ids:
        raise ValueError("provide --object-id or --manifest")
    return object_ids


def has_required_mesh(path):
    coacd = path / "coacd"
    return (
        (coacd / "coacd_1.urdf").is_file()
        and (coacd / "decomposed.obj").is_file()
        and any(coacd.glob("coacd_convex_piece_*.obj"))
    )


def main():
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    for object_id in requested_object_ids(args):
        source = source_root / object_id
        target = target_root / object_id
        if not has_required_mesh(source):
            raise FileNotFoundError("Incomplete source mesh: {}".format(source))
        if target.is_symlink():
            if target.resolve() != source:
                raise FileExistsError(
                    "Existing symlink points elsewhere: {}".format(target))
            print("[PASS] already staged: {}".format(object_id))
            continue
        if target.exists():
            if not target.is_dir() or not has_required_mesh(target):
                raise FileExistsError(
                    "Existing target is not a complete mesh: {}".format(target))
            print("[PASS] existing complete mesh: {}".format(object_id))
            continue
        target.symlink_to(source, target_is_directory=True)
        print("[PASS] staged symlink: {} -> {}".format(object_id, source))

    print("MESH_STAGE_RESULT=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
