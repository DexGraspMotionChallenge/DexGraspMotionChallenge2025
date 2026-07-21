"""Render selected residual-policy cases in isolated simulator processes."""

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--residual-checkpoint")
    mode.add_argument("--zero-residual", action="store_true")
    parser.add_argument("--residual-config", required=True)
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--trajectory-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--capture-stride", type=int, default=2)
    parser.add_argument("--min-free-vram-mb", type=int, default=4500)
    cli = parser.parse_args()
    with Path(cli.cases).open(encoding="utf-8") as handle:
        cases = yaml.safe_load(handle)["cases"]
    output = Path(cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for number, case in enumerate(cases, 1):
        tag = "{:02d}_{}_{}_{}".format(
            number, case["category"], case["outcome"],
            case["trajectory_index"])
        case_dir = output / tag
        command = [sys.executable, str(REPO_ROOT / "custom_tools/evaluate_residual_ppo.py"),
                   "--object-id", case["object_id"],
                   "--trajectory-root", str(Path(cli.trajectory_root).resolve()),
                   "--trajectory-indices", str(case["trajectory_index"]),
                   "--residual-config", str(Path(cli.residual_config).resolve()),
                   "--bc-checkpoint", str(Path(cli.bc_checkpoint).resolve()),
                   "--capture-dir", str(case_dir),
                   "--capture-stride", str(cli.capture_stride),
                   "--min-free-vram-mb", str(cli.min_free_vram_mb),
                   "--output", str(output / (tag + ".yaml"))]
        if cli.zero_residual:
            command.append("--zero-residual")
        else:
            command.extend(["--residual-checkpoint",
                            str(Path(cli.residual_checkpoint).resolve())])
        print("render case {}/{}: {}".format(number, len(cases), tag), flush=True)
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    print("[PASS] rendered {} cases to {}".format(len(cases), output))


if __name__ == "__main__":
    main()
