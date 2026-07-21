"""Read-only readiness check for DexGrasp training experiments."""

import argparse
import importlib
from pathlib import Path
import subprocess

REQUIRED_TRAJECTORY_KEYS = {'obj_rotmat', 'obj_scale', 'grasp_seqs'}
PRECOMPUTED_FEATURE_KEYS = {'obs', 'vis_unscale_actions', 'hand_pcds', 'obj_pcds'}


def check(condition, label, detail):
    state = 'PASS' if condition else 'FAIL'
    print('[{}] {}: {}'.format(state, label, detail))
    return condition


def warn(label, detail):
    print('[WARN] {}: {}'.format(label, detail))


def validate_trajectory(path):
    data = np.load(str(path), allow_pickle=True).item()
    missing = REQUIRED_TRAJECTORY_KEYS - set(data)
    if missing:
        return False, 'missing keys {}'.format(sorted(missing)), False
    grasp_seqs = data['grasp_seqs']
    valid_shape = grasp_seqs.ndim == 3 and grasp_seqs.shape[-1] == 28
    finite = bool(np.isfinite(grasp_seqs).all())
    precomputed = PRECOMPUTED_FEATURE_KEYS.issubset(data)
    detail = 'shape={}, precomputed_features={}'.format(grasp_seqs.shape, precomputed)
    return valid_shape and finite, detail, precomputed


def has_mesh(mesh_root, object_code):
    coacd_dir = mesh_root / object_code / 'coacd'
    required = [coacd_dir / 'coacd_1.urdf', coacd_dir / 'decomposed.obj']
    return all(path.is_file() for path in required) and any(
        coacd_dir.glob('coacd_convex_piece_*.obj'))


def parse_args():
    parser = argparse.ArgumentParser(description='Check DexGrasp experiment readiness.')
    parser.add_argument('--min-train-objects', type=int, default=1)
    parser.add_argument(
        '--static-only', action='store_true',
        help='Skip CUDA/dependency imports when another GPU process is running.')
    return parser.parse_args()


def main():
    global np
    args = parse_args()
    import numpy as np
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = repo_root / 'dexgrasp' / 'dataset'
    mesh_root = repo_root / 'assets' / 'meshdata'
    passed = True

    official_files = [
        'ActionDiffusion/bc/config/lhm_bc.yaml',
        'ActionDiffusion/bc/dataset/graspm3_dexrep.py',
        'dexgrasp/bc_env_infer.py',
        'dexgrasp/cfg/shadow_hand_grasp_dexrep_ijrr.yaml',
        'dexgrasp/data_preprocess.py',
        'dexgrasp/tasks/shadow_hand_grasp_dexrep_ijrr.py',
        'dexgrasp/train_bc_lighting_dexrep.py',
        'dexgrasp/utils/config.py',
        'dexgrasp/utils/info_summary_print.py',
        'dexgrasp/utils/test_env.py',
    ]
    official_clean = subprocess.run(
        ['git', 'diff', '--quiet', '--'] + official_files,
        cwd=str(repo_root),
        check=False,
    ).returncode == 0
    passed &= check(
        official_clean,
        'Official baseline files',
        'match Git baseline' if official_clean else 'contain local modifications',
    )

    custom_scripts = [
        'preprocess_graspm3.py', 'train_bc.py', 'evaluate_bc.py',
        'diagnose_graspm3_replay.py', 'export_experiment_curves.py',
    ]
    custom_tools_ok = all(
        (repo_root / 'custom_tools' / name).is_file() for name in custom_scripts)
    passed &= check(custom_tools_ok, 'Custom tool separation',
                    'all custom entry points are under custom_tools/')

    if args.static_only:
        warn('Runtime imports', 'skipped to avoid touching the occupied GPU')
    else:
        import isaacgym  # Isaac Gym must be imported before torch.  # noqa: F401
        import torch

        dependency_names = [
            'torchsdf', 'pytorch3d', 'pytorch_lightning', 'cv2', 'tensorboard']
        missing_dependencies = []
        for dependency in dependency_names:
            try:
                importlib.import_module(dependency)
            except Exception as error:
                missing_dependencies.append('{} ({})'.format(dependency, error))
        passed &= check(not missing_dependencies, 'Python dependencies',
                        'all required imports work' if not missing_dependencies
                        else '; '.join(missing_dependencies))

        cuda_ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
        cuda_detail = '{}; torch={}; cuda={}'.format(
            torch.cuda.get_device_name(0) if cuda_ok else 'CUDA unavailable',
            torch.__version__, torch.version.cuda)
        passed &= check(cuda_ok, 'CUDA', cuda_detail)

    object_sets = {}
    for split in ('train', 'valid'):
        files = sorted((dataset_root / split).glob('*.npy'))
        object_sets[split] = {path.stem for path in files}
        valid_files = 0
        precomputed_files = 0
        for path in files:
            valid, detail, precomputed = validate_trajectory(path)
            valid_files += int(valid)
            precomputed_files += int(precomputed)
            check(valid, '{} trajectory {}'.format(split, path.stem), detail)
        split_ok = valid_files == len(files) and bool(files)
        passed &= check(split_ok, '{} trajectory set'.format(split),
                        '{} files; {} with precomputed features'.format(
                            len(files), precomputed_files))

    mesh_objects = {path.name for path in mesh_root.iterdir() if path.is_dir()}
    complete_train_objects = sorted(
        code for code in object_sets['train'] if code in mesh_objects and has_mesh(mesh_root, code))
    requested_objects_ok = len(complete_train_objects) >= args.min_train_objects
    passed &= check(requested_objects_ok, 'Train trajectory-mesh pairs',
                    '{} complete; {} required; objects={}'.format(
                        len(complete_train_objects), args.min_train_objects,
                        complete_train_objects))

    incomplete_valid = sorted(code for code in object_sets['valid'] if not has_mesh(mesh_root, code))
    if incomplete_valid:
        warn('Validation meshes', 'missing for {}'.format(incomplete_valid))
    else:
        check(True, 'Validation meshes', 'complete')

    official_checkpoint = (
        repo_root / 'ActionDiffusion' / 'bc' / 'saved_models' /
        '1obj_seq2000_DexRep_pro100_start_uniform_vis_action_dsam_mod' / 'last.ckpt')
    passed &= check(official_checkpoint.is_file(), 'Official checkpoint', official_checkpoint)

    curve_dir = repo_root / 'dexgrasp' / 'results' / 'preparation_curve_verified'
    curve_files = ['training_scalars.csv', 'training_loss.png',
                   'evaluation_metrics.csv', 'evaluation_metrics.png']
    curves_ok = all((curve_dir / name).is_file() for name in curve_files)
    passed &= check(curves_ok, 'Curve export smoke', curve_dir)

    success_dir = repo_root / 'dexgrasp' / 'results' / 'render_success_batch40_env18_verified'
    failure_dir = repo_root / 'dexgrasp' / 'results' / 'render_test_failure_traj0'
    render_ok = (
        (success_dir / 'env018.mp4').is_file()
        and (success_dir / 'env018_success.png').is_file()
        and (failure_dir / 'env000.mp4').is_file()
        and (failure_dir / 'env000_final.png').is_file()
    )
    passed &= check(render_ok, 'Render export smoke',
                    'success and failure PNG/MP4 artifacts')

    ready_label = 'STATIC_READY' if args.static_only else 'READY'
    print('PREFLIGHT_RESULT={}'.format(ready_label if passed else 'NOT_READY'))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
