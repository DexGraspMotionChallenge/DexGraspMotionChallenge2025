"""Export report-ready curves and reward-term summaries for residual PPO."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REWARD_TERMS = (
    "approach", "contact", "lift", "milestone", "success_bonus",
    "failure_penalty", "residual_penalty", "smoothness_penalty")


def rolling_mean(values, window):
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(sum(values[start:index + 1]) / (index - start + 1))
    return result


def run(run_dir, output_dir):
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.csv").open(encoding="utf-8") as handle:
        training = list(csv.DictReader(handle))
    with (run_dir / "validation_metrics.csv").open(encoding="utf-8") as handle:
        validation = list(csv.DictReader(handle))

    iterations = [int(row["iteration"]) for row in training]
    rewards = [float(row["reward_reward_mean"]) for row in training]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(iterations, rewards, color="tab:blue", alpha=0.25,
              linewidth=1.0, label="Per-iteration mean")
    axis.plot(iterations, rolling_mean(rewards, 10), color="tab:blue",
              linewidth=2.0, label="10-iteration moving mean")
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("PPO iteration")
    axis.set_ylabel("Custom training reward")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    reward_plot = output_dir / "training_reward_curve.png"
    figure.savefig(reward_plot, dpi=180)
    plt.close(figure)

    validation_iterations = [int(row["iteration"]) for row in validation]
    success = [
        100.0 * float(row["macro_official_peak_success_rate"])
        for row in validation]
    lift = [
        100.0 * float(row["macro_mean_maximum_lift_m"])
        for row in validation]
    figure, success_axis = plt.subplots(figsize=(7.2, 4.2))
    lift_axis = success_axis.twinx()
    success_axis.plot(validation_iterations, success, marker="o",
                      linewidth=2.0, color="tab:green", label="Official success")
    lift_axis.plot(validation_iterations, lift, marker="s", linewidth=2.0,
                   color="tab:orange", label="Mean maximum lift")
    success_axis.set_xlabel("PPO iteration")
    success_axis.set_ylabel("Category-macro official success (%)",
                            color="tab:green")
    lift_axis.set_ylabel("Category-macro mean maximum lift (cm)",
                         color="tab:orange")
    success_axis.grid(alpha=0.25)
    success_axis.legend(loc="upper left")
    lift_axis.legend(loc="upper right")
    figure.tight_layout()
    validation_plot = output_dir / "validation_success_lift_curve.png"
    figure.savefig(validation_plot, dpi=180)
    plt.close(figure)

    summary_path = output_dir / "reward_term_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "term", "mean", "mean_step_std", "mean_absolute_fraction"))
        writer.writeheader()
        for term in REWARD_TERMS:
            writer.writerow({
                "term": term,
                "mean": sum(float(row["reward_{}_mean".format(term)])
                            for row in training) / len(training),
                "mean_step_std": sum(float(row["reward_{}_std".format(term)])
                                     for row in training) / len(training),
                "mean_absolute_fraction": sum(float(row[
                    "reward_{}_absolute_fraction".format(term)])
                    for row in training) / len(training),
            })
    for path in (reward_plot, validation_plot, summary_path):
        print(path)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    destination = args.output_dir or str(Path(args.run_dir) / "plots")
    run(args.run_dir, destination)
