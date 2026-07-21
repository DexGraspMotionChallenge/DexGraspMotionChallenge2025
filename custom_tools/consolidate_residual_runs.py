"""Merge resumed PPO segments and isolated validation results for plotting."""

import argparse
import csv
from pathlib import Path

import yaml


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segment", action="append", required=True,
        help="Run directory; later segments replace duplicate iterations.")
    parser.add_argument(
        "--validation", action="append", required=True,
        help="ITERATION:validation_metrics.csv or isolated aggregate YAML")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def validation_row(iteration, path):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        matches = [row for row in rows if int(row["iteration"]) == iteration]
        if len(matches) != 1:
            raise ValueError(
                "Expected one validation row for iteration {} in {}".format(
                    iteration, path))
        source = matches[0]
    else:
        with path.open(encoding="utf-8") as handle:
            source = yaml.safe_load(handle)
    return {
        "iteration": iteration,
        "global_step": iteration * 1024,
        "macro_official_peak_success_rate": source[
            "macro_official_peak_success_rate"],
        "macro_mean_maximum_lift_m": source[
            "macro_mean_maximum_lift_m"],
        "macro_failure_rate": source["macro_failure_rate"],
        "total_success_count": source["total_success_count"],
        "total_trajectory_count": source["total_trajectory_count"],
    }


def run(cli):
    output_dir = Path(cli.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    rows_by_iteration = {}
    fieldnames = None
    replacements = []
    segment_paths = []
    for segment in cli.segment:
        segment_path = Path(segment).expanduser().resolve()
        segment_paths.append(str(segment_path))
        with (segment_path / "metrics.csv").open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError("Segment metric columns differ")
            for row in reader:
                iteration = int(row["iteration"])
                if iteration in rows_by_iteration:
                    replacements.append(iteration)
                rows_by_iteration[iteration] = row
    iterations = sorted(rows_by_iteration)
    expected = list(range(iterations[0], iterations[-1] + 1))
    if iterations != expected:
        raise ValueError("Merged iterations are not contiguous")
    with (output_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_by_iteration[i] for i in iterations)

    validations = []
    validation_sources = []
    for specification in cli.validation:
        iteration_text, path_text = specification.split(":", 1)
        iteration = int(iteration_text)
        validations.append(validation_row(iteration, path_text))
        validation_sources.append({
            "iteration": iteration,
            "path": str(Path(path_text).expanduser().resolve()),
        })
    validations.sort(key=lambda item: item["iteration"])
    with (output_dir / "validation_metrics.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(validations[0]))
        writer.writeheader()
        writer.writerows(validations)

    manifest = {
        "segments_in_precedence_order": segment_paths,
        "duplicate_iterations_replaced_by_later_segments": sorted(
            set(replacements)),
        "merged_iteration_start": iterations[0],
        "merged_iteration_end": iterations[-1],
        "merged_iteration_count": len(iterations),
        "validation_sources": validation_sources,
    }
    with (output_dir / "consolidation_manifest.yaml").open(
            "w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    print(yaml.safe_dump(manifest, sort_keys=False))


if __name__ == "__main__":
    run(parse_cli())
