import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import torch
import pathlib

from functools import partial
import pytorch_lightning as pl

from  ActionDiffusion.model.policy.dp3_lightning import DP3Lightning
from ActionDiffusion.bc.model.policy.lhm_policy import LitDP3Model
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader
from ActionDiffusion.bc.dataset.graspm3_dexrep import GraspM3DexRepDataset
class BCTrainer:
    def __init__(self, args, env_args, train_loader=None,test_loader=None):
        self.args = args
        self.env_args = env_args
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.bc_dp3_model = LitDP3Model(args,env_args)
        a=1

    def train(self, ckpt_path=None):

        callback = ModelCheckpoint(dirpath=self.args.exp_dir, filename='{step}',
                                   save_top_k=-1, save_last=True, every_n_train_steps=1000)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [callback, lr_monitor]
        trainer = pl.Trainer(accelerator='gpu', devices=-1, precision=32, max_epochs=self.args.num_epochs,
                             callbacks=callbacks, log_every_n_steps=2, check_val_every_n_epoch=5,
                             default_root_dir=os.path.join(self.args.exp_dir, "tensorboard_logs"))

        trainer.fit(model=self.bc_dp3_model, train_dataloaders=self.train_loader, ckpt_path=ckpt_path, val_dataloaders=self.test_loader)


def main(args, env_args):
    args.obs_type='dexrep'

    kstr = 'sim_action' if args.use_sim_action else 'vis_action'
    args.horizon=8
    args.n_action_steps=7

    args.task_name = '1obj_seq2000_DexRep_pro100_start_uniform_DP3Dexrep_horizon{}_use_smooth_{}_dsam_mod'.format(args.horizon, kstr)

    if  args.policy.actor_critic=='ActorCriticDexRep':
        args.policy.dexrep_encoder_cfg.env_cfg.obs_dim.pop('pnG')
    elif args.policy.actor_critic=='ActorCriticPNG':
        args.policy.dexrep_encoder_cfg.env_cfg.obs_dim.pop('dexrep_sensor')
        args.policy.dexrep_encoder_cfg.env_cfg.obs_dim.pop('dexrep_pnl')

    args.seq_num=100
    args.add_noise=False
    args.obs_as_dexrep_cond=True
    args.obs_as_global_cond=False
    args.obs_type='dexrep'

    # args.noise_val=0.02

    args.exp_dir = os.path.join(args.exp_dir, args.task_name)
    os.makedirs(args.exp_dir,exist_ok=True)

    ds_train = GraspM3DexRepDataset(args, ds_name='train')
    ds_test = GraspM3DexRepDataset(args, ds_name='test')
    my_collate_fn = partial(ds_train.collate_fn, horizon=args.horizon)


    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=my_collate_fn) #**args.dataloader
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=my_collate_fn)

    trainer = BCTrainer(args,env_args, train_loader, test_loader)
    trainer.train()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load("{}/dp3.yaml".format('../ActionDiffusion/bc/config'))
    env_args = OmegaConf.load("{}/shadow_hand_grasp_dexrep_ijrr.yaml".format('./cfg'))
    main(args, env_args)
