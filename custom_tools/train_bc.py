"""Train a BC policy without changing the challenge's official trainer.

Run this script from any directory.  Dataset-relative paths are resolved by
temporarily using ``dexgrasp/`` as the working directory, matching the layout
assumed by the original project.
"""

import argparse
import collections
import hashlib
import json
import os
import pathlib
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEXGRASP_ROOT = REPO_ROOT / "dexgrasp"
for import_root in (str(REPO_ROOT), str(DEXGRASP_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import isaacgym  # Isaac Gym must be imported before torch.  # noqa: E402,F401
import torch  # noqa: E402


import pytorch_lightning as pl
from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from custom_tools.graspm3_dexrep_dataset import GraspM3DexRepDataset


def require_free_vram(min_free_vram_mb):
    """Stop before allocating a model when another GPU job leaves too little room."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; BC training requires the NVIDIA GPU.")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_mb = free_bytes / (1024 ** 2)
    total_mb = total_bytes / (1024 ** 2)
    print("GPU memory before training: {:.0f}/{:.0f} MiB free".format(free_mb, total_mb))
    if free_mb < min_free_vram_mb:
        raise RuntimeError(
            "Only {:.0f} MiB VRAM is free, below the safety threshold of {} MiB. "
            "Wait for the other GPU process to finish instead of competing for memory."
            .format(free_mb, min_free_vram_mb)
        )


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_category_balanced_sampler(dataset, seed, online_sample_fraction=None):
    """Give bottle/mug/bowl/camera equal expected sampling probability."""
    if hasattr(dataset, 'sample_categories'):
        categories = np.asarray(dataset.sample_categories)
    else:
        sequence_object_indices = dataset.data['obj_code_idx']
        categories = []
        for object_index in sequence_object_indices:
            object_id = dataset.obj_code_name_list[int(object_index)]
            parts = object_id.split('-', 2)
            if len(parts) < 3:
                raise ValueError('Cannot infer category from object ID: {}'.format(object_id))
            categories.append(parts[1])
        if dataset.is_flat:
            categories = np.repeat(np.asarray(categories), dataset.num_frame)
    category_list = categories.tolist() if isinstance(categories, np.ndarray) else categories
    counts = collections.Counter(category_list)
    if online_sample_fraction is not None:
        if not hasattr(dataset, 'sample_sources'):
            raise ValueError('Online source balancing requires an augmented dataset')
        fraction = float(online_sample_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError('online_sample_fraction must be in (0, 1)')
        sources = np.asarray(dataset.sample_sources)
        if len(sources) != len(category_list):
            raise ValueError('Sample source/category lengths differ')
        group_counts = collections.Counter(zip(sources.tolist(), category_list))
        source_targets = {0: 1.0 - fraction, 1: fraction}
        category_count = len(counts)
        weights = torch.as_tensor([
            source_targets[int(source)]
            / category_count
            / group_counts[(int(source), category)]
            for source, category in zip(sources, category_list)
        ], dtype=torch.double)
        print('Balanced sampler target online fraction: {:.3f}'.format(fraction))
    else:
        weights = torch.as_tensor(
            [1.0 / counts[category] for category in category_list], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    print('Balanced sampler frame counts before weighting: {}'.format(dict(sorted(counts.items()))))
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=generator)


class OnlineAugmentedDataset(Dataset):
    """Append student-visited states labeled by the routed teacher.

    Offline samples retain the 70:30 teacher/demo target.  An online sample
    has no demonstration action, so both targets are set to the teacher action;
    its effective supervision is therefore 100% teacher without changing the
    loss implementation.
    """

    def __init__(self, offline_dataset, online_path):
        self.offline = offline_dataset
        online = np.load(online_path, allow_pickle=False)
        self.online_observations = online['observations'].astype(np.float32, copy=False)
        self.online_actions = online['teacher_actions'].astype(np.float32, copy=False)
        self.online_category_indices = online['category_indices'].astype(np.int64, copy=False)
        if len(self.online_observations) != len(self.online_actions):
            raise ValueError('Online observations/actions are not aligned')
        if self.online_observations.shape[1:] != self.offline.data['obs'].shape[1:]:
            raise ValueError(
                'Online observation shape {} does not match offline {}'.format(
                    self.online_observations.shape[1:],
                    self.offline.data['obs'].shape[1:]))
        category_names = np.asarray(['bottle', 'mug', 'bowl', 'camera'])
        if np.any(self.online_category_indices < 0) or np.any(
                self.online_category_indices >= len(category_names)):
            raise ValueError('Invalid online category index')
        sequence_categories = []
        for object_index in self.offline.data['obj_code_idx']:
            object_id = self.offline.obj_code_name_list[int(object_index)]
            sequence_categories.append(object_id.split('-', 2)[1])
        offline_categories = np.asarray(sequence_categories)
        if self.offline.is_flat:
            offline_categories = np.repeat(
                offline_categories, self.offline.num_frame)
        self.sample_categories = np.concatenate([
            offline_categories,
            category_names[self.online_category_indices],
        ])
        self.sample_sources = np.concatenate([
            np.zeros(len(self.offline), dtype=np.int8),
            np.ones(len(self.online_observations), dtype=np.int8),
        ])
        print('Online aggregation: offline={} online={} total={}'.format(
            len(self.offline), len(self.online_observations), len(self)))

    def __len__(self):
        return len(self.offline) + len(self.online_observations)

    def __getitem__(self, index):
        if index < len(self.offline):
            return self.offline[index]
        online_index = index - len(self.offline)
        observation = self.online_observations[online_index].copy()
        if self.offline.args.add_noise:
            noise = np.random.uniform(
                -self.offline.args.noise_val,
                self.offline.args.noise_val,
                size=self.offline.pro_dim).astype(np.float32)
            observation[..., :self.offline.pro_dim] += noise
        action = self.online_actions[online_index]
        return {
            'obs': observation,
            'actions': action,
            'teacher_actions': action,
            'obj_code_idx': np.int64(-1),
            'sample_index': np.int64(online_index),
        }


class DistillationBCModel(LitBCModel):
    """Unified student trained against routed teachers and original demos."""

    def __init__(self, args, env_args):
        super().__init__(args, env_args)
        config = args.distillation
        self.teacher_weight = float(config.teacher_weight)
        self.demo_weight = float(config.demo_weight)
        if self.teacher_weight < 0 or self.demo_weight < 0:
            raise ValueError('Distillation weights must be non-negative')
        if abs(self.teacher_weight + self.demo_weight - 1.0) > 1e-6:
            raise ValueError('Distillation weights must sum to one')

    def training_step(self, batch, batch_idx):
        prediction = self.forward(batch)
        teacher = self.cal_loss(prediction, batch['teacher_actions'])
        demo = self.cal_loss(prediction, batch['actions'])
        loss = (self.teacher_weight * teacher['loss']
                + self.demo_weight * demo['loss'])
        self.log_dict({
            'train_loss': loss,
            'teacher_loss': teacher['loss'],
            'demo_loss': demo['loss'],
            'teacher_wrist_loss': teacher['wrist_loss'],
            'teacher_ori_loss': teacher['ori_loss'],
            'teacher_finger_loss': teacher['finger_loss'],
        }, prog_bar=True, on_epoch=True)
        return loss


class BCTrainer:
    def __init__(self, args,env_args,  train_loader=None,test_loader=None,
                 init_checkpoint=None):
        self.args = args
        self.env_args = env_args
        self.train_loader = train_loader
        self.test_loader = test_loader

        if args.get('distillation', {}).get('enabled', False):
            self.bc_model = DistillationBCModel(args, env_args.env)
        else:
            self.bc_model = LitBCModel(args, env_args.env)
        if init_checkpoint is not None:
            checkpoint = torch.load(init_checkpoint, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.bc_model.load_state_dict(state_dict, strict=True)
            print(
                'Initialized model weights from {} (epoch={}, optimizer state not loaded)'
                .format(init_checkpoint, checkpoint.get('epoch', 'unknown')))

    def train(self, ckpt_path=None):

        callback = ModelCheckpoint(
            dirpath=self.args.exp_dir,
            filename='{epoch:03d}-{step}',
            save_top_k=-1,
            save_last=True,
            every_n_epochs=self.args.get('checkpoint_every_n_epochs', 1),
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [callback, lr_monitor]
        trainer_kwargs = dict(
            accelerator='gpu', devices=1, precision=32, max_epochs=self.args.num_epochs,
            callbacks=callbacks, log_every_n_steps=5,
            check_val_every_n_epoch=self.args.get('check_val_every_n_epoch', 1),
            default_root_dir=os.path.join(self.args.exp_dir, "tensorboard_logs"))
        if self.args.get('limit_train_batches') is not None:
            trainer_kwargs['limit_train_batches'] = int(self.args.limit_train_batches)
        if self.args.get('limit_val_batches') is not None:
            trainer_kwargs['limit_val_batches'] = int(self.args.limit_val_batches)
        trainer = pl.Trainer(**trainer_kwargs)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        trainer.fit(model=self.bc_model, train_dataloaders=self.train_loader,
                    ckpt_path=ckpt_path, val_dataloaders=self.test_loader)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        resource_summary = {
            'elapsed_seconds': float(elapsed),
            'global_step': int(trainer.global_step),
            'peak_allocated_mib': float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            'peak_reserved_mib': float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
            'free_vram_after_mib': float(free_bytes / (1024 ** 2)),
            'total_vram_mib': float(total_bytes / (1024 ** 2)),
        }
        from omegaconf import OmegaConf
        OmegaConf.save(
            OmegaConf.create(resource_summary),
            os.path.join(self.args.exp_dir, 'resource_summary.yaml'))
        print(
            'Training resource summary: elapsed={:.2f}s, peak_allocated={:.0f} MiB, '
            'peak_reserved={:.0f} MiB'.format(
                resource_summary['elapsed_seconds'],
                resource_summary['peak_allocated_mib'],
                resource_summary['peak_reserved_mib']))


def main(args, env_args, resume_checkpoint=None, init_checkpoint=None,
         min_free_vram_mb=5000):

    require_free_vram(min_free_vram_mb)

    seed = int(args.get('seed', 0))
    pl.seed_everything(0 if seed < 0 else seed, workers=True)

    kstr = 'sim_action' if args.use_sim_action else 'vis_action'

    default_run_name = '1obj_seq2000_DexRep_pro100_start_uniform_{}_dsam_mod'.format(kstr)
    args.task_name = args.get('run_name', default_run_name)
    args.policy.actor_critic = 'ActorCriticDexRep'
    env_args.env.obs_dim.pop('pnG')

    # Keep historical defaults, but allow independent custom configs to turn
    # observation augmentation on for controlled experiments.
    if 'add_noise' not in args:
        args.add_noise = False
    if 'noise_val' not in args:
        args.noise_val = 0.02

    args.exp_dir = os.path.abspath(os.path.join(args.exp_dir, args.task_name))
    existing_checkpoints = list(pathlib.Path(args.exp_dir).glob('*.ckpt'))
    if existing_checkpoints and resume_checkpoint is None:
        raise FileExistsError(
            'Run directory already contains checkpoints: {}. '
            'Choose a new --run-name or pass --resume-checkpoint.'.format(args.exp_dir)
        )
    if resume_checkpoint is not None:
        resume_checkpoint = os.path.abspath(os.path.expanduser(resume_checkpoint))
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError('Resume checkpoint not found: {}'.format(resume_checkpoint))
    if init_checkpoint is not None and not os.path.isfile(init_checkpoint):
        raise FileNotFoundError('Initialization checkpoint not found: {}'.format(init_checkpoint))

    os.makedirs(args.exp_dir, exist_ok=True)

    from omegaconf import OmegaConf
    OmegaConf.save(args, os.path.join(args.exp_dir, 'resolved_config.yaml'))
    OmegaConf.save(env_args, os.path.join(args.exp_dir, 'resolved_env_config.yaml'))
    metadata = OmegaConf.create({
        'started_at': datetime.now().isoformat(timespec='seconds'),
        'hostname': socket.gethostname(),
        'python': sys.version.split()[0],
        'command': ' '.join(sys.argv),
        'resume_checkpoint': resume_checkpoint,
        'init_checkpoint': init_checkpoint,
        'init_checkpoint_sha256': (
            checkpoint_sha256(init_checkpoint) if init_checkpoint is not None else None),
    })
    OmegaConf.save(metadata, os.path.join(args.exp_dir, 'run_metadata.yaml'))

    ds_train = GraspM3DexRepDataset(args, ds_name='train')
    ds_test = GraspM3DexRepDataset(args, ds_name='test')

    online_action_file = args.get('distillation', {}).get('online_action_file')
    if online_action_file:
        online_path = pathlib.Path(str(online_action_file)).expanduser()
        if not online_path.is_absolute():
            online_path = pathlib.Path.cwd() / online_path
        online_path = online_path.resolve()
        if not online_path.is_file():
            raise FileNotFoundError('Online aggregation file: {}'.format(online_path))
        ds_train = OnlineAugmentedDataset(ds_train, str(online_path))

    sampler = None
    if args.get('category_balanced_sampling', False):
        sampler = build_category_balanced_sampler(
            ds_train, seed, args.get('online_sample_fraction'))
    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=sampler is None,
        sampler=sampler, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, drop_last=True,num_workers=4, pin_memory=True)

    trainer = BCTrainer(
        args, env_args, train_loader, test_loader,
        init_checkpoint=init_checkpoint)
    trainer.train(ckpt_path=resume_checkpoint)


def parse_cli():
    parser = argparse.ArgumentParser(description='Train the DexRep behavior-cloning baseline.')
    parser.add_argument('--config', default=str(REPO_ROOT / 'ActionDiffusion/bc/config/lhm_bc.yaml'))
    parser.add_argument('--env-config', default=str(DEXGRASP_ROOT / 'cfg/shadow_hand_grasp_dexrep_ijrr.yaml'))
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--teacher-weight', type=float, default=None)
    parser.add_argument('--online-sample-fraction', type=float, default=None)
    parser.add_argument('--noise-value', type=float, default=None)
    parser.add_argument('--seq-num', type=int, default=None)
    parser.add_argument('--val-seq-num', type=int, default=None)
    parser.add_argument(
        '--train-category', choices=('bottle', 'mug', 'bowl', 'camera'),
        default=None,
        help='Train and validate only on one category from the frozen manifest.')
    parser.add_argument('--category-manifest', default=str(
        REPO_ROOT / 'custom_tools/configs/object_split_final.json'))
    parser.add_argument('--resume-checkpoint', default=None)
    parser.add_argument(
        '--init-checkpoint', default=None,
        help='Load model weights only and start a new run at epoch 0.')
    parser.add_argument(
        '--min-free-vram-mb', type=int, default=5000,
        help='Abort before training if less VRAM is free (default: 5000 MiB).')
    parser.add_argument('--print-config', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    from omegaconf import OmegaConf

    cli = parse_cli()
    if cli.resume_checkpoint is not None and cli.init_checkpoint is not None:
        raise ValueError('--resume-checkpoint and --init-checkpoint are mutually exclusive')
    resume_checkpoint = (
        str(Path(cli.resume_checkpoint).expanduser().resolve())
        if cli.resume_checkpoint is not None else None)
    init_checkpoint = (
        str(Path(cli.init_checkpoint).expanduser().resolve())
        if cli.init_checkpoint is not None else None)
    args = OmegaConf.load(str(Path(cli.config).expanduser().resolve()))
    env_args = OmegaConf.load(str(Path(cli.env_config).expanduser().resolve()))
    for cli_name, config_name in (
        ('run_name', 'run_name'),
        ('seed', 'seed'),
        ('num_epochs', 'num_epochs'),
        ('batch_size', 'batch_size'),
        ('learning_rate', 'lr'),
        ('online_sample_fraction', 'online_sample_fraction'),
        ('noise_value', 'noise_val'),
        ('seq_num', 'seq_num'),
        ('val_seq_num', 'val_seq_num'),
    ):
        value = getattr(cli, cli_name)
        if value is not None:
            OmegaConf.update(args, config_name, value)

    if cli.teacher_weight is not None:
        if not 0.0 <= cli.teacher_weight <= 1.0:
            raise ValueError('--teacher-weight must be in [0, 1]')
        OmegaConf.update(args, 'distillation.teacher_weight', cli.teacher_weight)
        OmegaConf.update(args, 'distillation.demo_weight', 1.0 - cli.teacher_weight)

    if cli.train_category is not None:
        manifest_path = Path(cli.category_manifest).expanduser().resolve()
        with manifest_path.open('r', encoding='utf-8') as handle:
            manifest = json.load(handle)
        object_ids = manifest['categories'][cli.train_category]['train']
        if len(object_ids) != 4:
            raise ValueError(
                'Expected four frozen training objects for {}, got {}'.format(
                    cli.train_category, len(object_ids)))
        OmegaConf.update(args, 'train_obj_code_list', object_ids)
        OmegaConf.update(args, 'val_obj_code_list', object_ids)
        OmegaConf.update(args, 'expert_category', cli.train_category)
        OmegaConf.update(args, 'category_manifest', str(manifest_path))

    if cli.print_config:
        print(OmegaConf.to_yaml(args, resolve=False))
    else:
        original_cwd = Path.cwd()
        try:
            os.chdir(str(DEXGRASP_ROOT))
            main(
                args,
                env_args,
                resume_checkpoint=resume_checkpoint,
                init_checkpoint=init_checkpoint,
                min_free_vram_mb=cli.min_free_vram_mb,
            )
        finally:
            os.chdir(str(original_cwd))


