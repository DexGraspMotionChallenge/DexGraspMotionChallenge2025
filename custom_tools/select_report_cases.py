"""Choose one success and one representative failure for each category."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    with Path(cli.evaluation).open(encoding="utf-8") as handle:
        evaluation = yaml.safe_load(handle)
    by_category = {}
    for obj in evaluation["objects"]:
        category = obj["object_id"].split("-", 2)[1]
        successes = set(obj["official_peak_success_source_indices"])
        records = list(zip(obj["trajectory_indices"],
                           obj["diagnostic_maximum_lift_m_by_trajectory"]))
        entry = by_category.setdefault(category, {"success": [], "failure": []})
        for index, lift in records:
            outcome = "success" if index in successes else "failure"
            entry[outcome].append({"object_id": obj["object_id"],
                                   "trajectory_index": int(index),
                                   "maximum_lift_m": float(lift),
                                   "outcome": outcome,
                                   "category": category})
    cases = []
    for category in sorted(by_category):
        successful = by_category[category]["success"]
        failures = [item for item in by_category[category]["failure"]
                    if -0.03 <= item["maximum_lift_m"] <= 0.30]
        if not successful or not failures:
            raise ValueError("{} lacks a success or valid failure".format(category))
        cases.append(max(successful, key=lambda item: item["maximum_lift_m"]))
        cases.append(max(failures, key=lambda item: item["maximum_lift_m"]))
    result = {"source_evaluation": str(Path(cli.evaluation).resolve()),
              "selection_rule": "Highest-lift success and highest-lift valid failure per category",
              "cases": cases}
    output = Path(cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, allow_unicode=True, sort_keys=False)
    print("[PASS] selected {} report cases: {}".format(len(cases), output))


if __name__ == "__main__":
    main()
