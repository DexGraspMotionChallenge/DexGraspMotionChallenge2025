"""Export report figures for noise BC, ungated PPO and gated PPO."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


TERMS = ("approach", "contact", "lift", "milestone", "success_bonus",
         "failure_penalty", "residual_penalty", "smoothness_penalty",
         "gate_penalty")


def rolling(values, window=10):
    return [sum(values[max(0, i - window + 1):i + 1])
            / len(values[max(0, i - window + 1):i + 1])
            for i in range(len(values))]


def metrics(run_dir):
    with (Path(run_dir) / "metrics.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluations(paths):
    values = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            item = yaml.safe_load(handle)
        checkpoint = Path(item["residual_checkpoint"]).stem
        values.append((int(checkpoint.split("_")[-1]), item))
    return sorted(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--gated-run", required=True)
    parser.add_argument("--baseline-eval", required=True)
    parser.add_argument("--control-eval", action="append", required=True)
    parser.add_argument("--gated-eval", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    cli = parser.parse_args()
    output = Path(cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    control = metrics(cli.control_run)
    gated = metrics(cli.gated_run)
    control_eval = evaluations(cli.control_eval)
    gated_eval = evaluations(cli.gated_eval)
    with Path(cli.baseline_eval).open(encoding="utf-8") as handle:
        baseline = yaml.safe_load(handle)

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for label, rows, color in (("Ungated residual", control, "tab:blue"),
                               ("Gated residual", gated, "tab:orange")):
        x = [int(row["iteration"]) for row in rows]
        y = rolling([float(row["reward_reward_mean"]) for row in rows])
        axis.plot(x, y, linewidth=2, label=label, color=color)
    axis.set(xlabel="PPO iteration", ylabel="10-iteration mean training reward")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "training_reward_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    baseline_success = 100 * baseline["macro_official_peak_success_rate"]
    axis.axhline(baseline_success, color="black", linestyle="--",
                 label="Noise BC baseline")
    for label, values, color in (("Ungated residual", control_eval, "tab:blue"),
                                 ("Gated residual", gated_eval, "tab:orange")):
        axis.plot([x for x, _ in values],
                  [100 * d["macro_official_peak_success_rate"] for _, d in values],
                  marker="o", linewidth=2, color=color, label=label)
    axis.set(xlabel="PPO iteration", ylabel="Category-macro official success (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "heldout_success_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.axhline(100 * baseline["macro_mean_maximum_lift_m"], color="black",
                 linestyle="--", label="Noise BC baseline")
    for label, values, color in (("Ungated residual", control_eval, "tab:blue"),
                                 ("Gated residual", gated_eval, "tab:orange")):
        axis.plot([x for x, _ in values],
                  [100 * d["macro_mean_maximum_lift_m"] for _, d in values],
                  marker="s", linewidth=2, color=color, label=label)
    axis.set(xlabel="PPO iteration", ylabel="Category-macro maximum lift (cm)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "heldout_lift_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    x = [int(row["iteration"]) for row in gated]
    axis.plot(x, rolling([float(row["wrist_gate_mean"]) for row in gated]),
              linewidth=2, label="Wrist gate")
    axis.plot(x, rolling([float(row["finger_gate_mean"]) for row in gated]),
              linewidth=2, label="Finger gate")
    axis.set(xlabel="PPO iteration", ylabel="10-iteration mean gate value", ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "gate_curve.png", dpi=180)
    plt.close(figure)

    term_summary = output / "reward_term_summary.csv"
    term_rows = []
    for method, rows in (("ungated", control), ("gated", gated)):
        for term in TERMS:
            term_rows.append({
                "method": method,
                "term": term,
                "mean": sum(float(row["reward_{}_mean".format(term)])
                            for row in rows) / len(rows),
                "mean_step_std": sum(float(row["reward_{}_std".format(term)])
                                     for row in rows) / len(rows),
                "mean_absolute_fraction": sum(float(row[
                    "reward_{}_absolute_fraction".format(term)])
                    for row in rows) / len(rows),
            })
    with term_summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=term_rows[0].keys())
        writer.writeheader()
        writer.writerows(term_rows)
    figure, axis = plt.subplots(figsize=(9.0, 4.5))
    positions = list(range(len(TERMS)))
    for offset, method, color in ((-0.18, "ungated", "tab:blue"),
                                  (0.18, "gated", "tab:orange")):
        values = [100 * next(row["mean_absolute_fraction"] for row in term_rows
                             if row["method"] == method and row["term"] == term)
                  for term in TERMS]
        axis.bar([x + offset for x in positions], values, width=0.36,
                 label=method.capitalize(), color=color)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [term.replace("_penalty", "") for term in TERMS],
        rotation=30, ha="right")
    axis.set_ylabel("Mean absolute reward contribution (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "reward_contribution_comparison.png", dpi=180)
    plt.close(figure)

    summary = output / "comparison_summary.csv"
    fields = ("method", "iteration", "overall_success", "macro_success",
              "macro_lift_m", "failure_rate")
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"method": "noise_bc", "iteration": 0,
                         "overall_success": baseline["overall_official_peak_success_rate"],
                         "macro_success": baseline["macro_official_peak_success_rate"],
                         "macro_lift_m": baseline["macro_mean_maximum_lift_m"],
                         "failure_rate": baseline["macro_failure_rate"]})
        for method, values in (("ungated", control_eval), ("gated", gated_eval)):
            for iteration, item in values:
                writer.writerow({"method": method, "iteration": iteration,
                                 "overall_success": item["overall_official_peak_success_rate"],
                                 "macro_success": item["macro_official_peak_success_rate"],
                                 "macro_lift_m": item["macro_mean_maximum_lift_m"],
                                 "failure_rate": item["macro_failure_rate"]})
    for path in sorted(output.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
