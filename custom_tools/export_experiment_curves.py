"""Export report-ready training and evaluation curves.

This script only reads TensorBoard events and evaluation YAML files. It does
not depend on Isaac Gym and can be rerun after every experiment.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def write_csv(path, fieldnames, rows):
    with path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_training(event_dir, output_dir):
    accumulator = EventAccumulator(str(event_dir), size_guidance={'scalars': 0})
    accumulator.Reload()
    tags = accumulator.Tags().get('scalars', [])
    if not tags:
        raise RuntimeError('No TensorBoard scalar data found in {}'.format(event_dir))

    rows = []
    scalar_data = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        scalar_data[tag] = events
        rows.extend({
            'tag': tag,
            'step': event.step,
            'wall_time': event.wall_time,
            'value': event.value,
        } for event in events)

    csv_path = output_dir / 'training_scalars.csv'
    write_csv(csv_path, ['tag', 'step', 'wall_time', 'value'], rows)

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    plotted = False
    for tag, style in (
        ('train_loss_step', {'alpha': 0.65, 'linewidth': 1.5}),
        ('train_loss_epoch', {'marker': 'o', 'linewidth': 2.0}),
        ('val_loss', {'marker': 's', 'linewidth': 2.0}),
    ):
        events = scalar_data.get(tag, [])
        if events:
            axis.plot([event.step for event in events],
                      [event.value for event in events], label=tag, **style)
            plotted = True
    if not plotted:
        raise RuntimeError('Expected loss tags were not found in {}'.format(event_dir))

    axis.set_xlabel('Optimizer step')
    axis.set_ylabel('Behavior-cloning loss')
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    plot_path = output_dir / 'training_loss.png'
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return csv_path, plot_path


def read_evaluation_rows(result_files):
    rows = []
    for result_file in result_files:
        with result_file.open('r', encoding='utf-8') as yaml_file:
            result = yaml.safe_load(yaml_file)
        if not isinstance(result, dict):
            continue
        if 'total_succ_rates' not in result or 'total_mean_rewards' not in result:
            continue
        rows.append({
            'result_file': str(result_file.resolve()),
            'result_tag': result.get('result_tag', ''),
            'checkpoint': result.get('checkpoint', ''),
            'checkpoint_sha256': result.get('checkpoint_sha256', ''),
            'checkpoint_epoch': result.get('checkpoint_epoch', ''),
            'checkpoint_global_step': result.get('checkpoint_global_step', ''),
            'seed': result.get('seed', ''),
            'dataset_name': result.get('dataset_name', ''),
            # Older YAML files predate explicit metric labels, so never guess
            # that they are challenge-comparable.
            'success_metric': result.get('success_metric', 'legacy_unspecified'),
            'reward_kind': result.get('reward_kind', 'legacy_unspecified'),
            'success_rate': float(result['total_succ_rates']),
            'mean_rollout_reward': float(result['total_mean_rewards']),
        })
    return rows


def export_evaluations(result_files, output_dir):
    rows = read_evaluation_rows(result_files)
    if not rows:
        raise RuntimeError('No valid evaluation metrics were found.')

    csv_path = output_dir / 'evaluation_metrics.csv'
    fieldnames = list(rows[0].keys())
    write_csv(csv_path, fieldnames, rows)

    steps = [row['checkpoint_global_step'] for row in rows]
    use_steps = all(step not in ('', None) for step in steps)
    x_values = [int(step) for step in steps] if use_steps else list(range(len(rows)))
    x_label = 'Checkpoint global step' if use_steps else 'Evaluation index'

    figure, success_axis = plt.subplots(figsize=(7.2, 4.2))
    reward_axis = success_axis.twinx()
    success_axis.plot(x_values, [100.0 * row['success_rate'] for row in rows],
                      color='tab:blue', marker='o', label='Success rate')
    reward_axis.plot(x_values, [row['mean_rollout_reward'] for row in rows],
                     color='tab:orange', marker='s', label='Mean rollout reward')
    success_axis.set_xlabel(x_label)
    success_metrics = sorted({row['success_metric'] for row in rows})
    reward_kinds = sorted({row['reward_kind'] for row in rows})
    success_axis.set_ylabel(
        'Success rate (%) [{}]'.format(', '.join(success_metrics)), color='tab:blue')
    reward_axis.set_ylabel(
        'Mean rollout reward [{}]'.format(', '.join(reward_kinds)), color='tab:orange')
    success_axis.grid(alpha=0.25)
    success_axis.legend(loc='upper left')
    reward_axis.legend(loc='upper right')
    if not use_steps:
        labels = [row['result_tag'] or Path(row['checkpoint']).parent.name for row in rows]
        success_axis.set_xticks(x_values)
        success_axis.set_xticklabels(labels, rotation=15, ha='right')
    figure.tight_layout()
    plot_path = output_dir / 'evaluation_metrics.png'
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return csv_path, plot_path


def parse_args():
    parser = argparse.ArgumentParser(description='Export DexGrasp experiment curves to CSV and PNG.')
    parser.add_argument('--event-dir', type=Path)
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--result-pattern', default='*.yaml')
    parser.add_argument('--result-file', action='append', type=Path, default=[])
    parser.add_argument('--output-dir', required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.event_dir is None and args.results_dir is None and not args.result_file:
        raise SystemExit('Provide --event-dir and/or evaluation result files.')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    if args.event_dir is not None:
        outputs.extend(export_training(args.event_dir, args.output_dir))

    result_files = list(args.result_file)
    if args.results_dir is not None:
        result_files.extend(sorted(args.results_dir.glob(args.result_pattern)))
    if result_files:
        outputs.extend(export_evaluations(result_files, args.output_dir))

    for output in outputs:
        print(output.resolve())


if __name__ == '__main__':
    main()
