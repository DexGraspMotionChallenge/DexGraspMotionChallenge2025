"""Build a fixed-size, state-aligned replay mixture from two online rounds."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


META_KEYS = ("category_indices", "object_indices", "trajectory_indices",
             "frame_indices", "object_ids")
DATA_KEYS = ("observations", "teacher_actions", "student_actions")


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round1", required=True)
    parser.add_argument("--round2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def main():
    cli = parse_cli()
    paths = [Path(cli.round1).expanduser().resolve(),
             Path(cli.round2).expanduser().resolve()]
    output = Path(cli.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    rounds = [np.load(path, allow_pickle=False) for path in paths]
    for key in META_KEYS:
        if not np.array_equal(rounds[0][key], rounds[1][key]):
            raise ValueError("Online rounds are not aligned for {}".format(key))
    count = len(rounds[0]["observations"])
    rng = np.random.RandomState(cli.seed)
    selected_round = rng.randint(0, 2, size=count).astype(np.int8)
    choose_round2 = selected_round.astype(bool)
    merged = {}
    for key in DATA_KEYS:
        if rounds[0][key].shape != rounds[1][key].shape:
            raise ValueError("Shape mismatch for {}".format(key))
        value = rounds[0][key].copy()
        value[choose_round2] = rounds[1][key][choose_round2]
        merged[key] = value
    for key in META_KEYS:
        merged[key] = rounds[0][key]
    merged["selected_round"] = selected_round + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "fixed-size state-aligned two-round replay mixture",
        "seed": cli.seed,
        "inputs": [str(path) for path in paths],
        "sample_count": count,
        "round1_samples": int((selected_round == 0).sum()),
        "round2_samples": int((selected_round == 1).sum()),
        "metadata_alignment_verified": True,
    }
    with output.with_suffix(".yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    print("ONLINE_ROUNDS_MERGED=COMPLETE samples={}".format(count), flush=True)


if __name__ == "__main__":
    main()
