import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import pathlib


import pytorch_lightning as pl
from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader
from ActionDiffusion.bc.dataset.graspm3_dexrep import GraspM3DexRepDataset
class BCTrainer:
    def __init__(self, args,env_args,  train_loader=None,test_loader=None):
        self.args = args
        self.env_args = env_args
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.bc_model = LitBCModel(args, env_args.env)
        # self.bc_model.load_state_dict(ckpt['state_dict'])
        # a=1

    def train(self, ckpt_path=None):

        callback = ModelCheckpoint(dirpath=self.args.exp_dir, filename='{step}',
                                   save_top_k=-1, save_last=True, every_n_train_steps=50000)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [callback, lr_monitor]
        trainer = pl.Trainer(accelerator='gpu', devices=-1, precision=32, max_epochs=self.args.num_epochs,
                             callbacks=callbacks, log_every_n_steps=5, check_val_every_n_epoch=5,
                             default_root_dir=os.path.join(self.args.exp_dir, "tensorboard_logs"))

        trainer.fit(model=self.bc_model, train_dataloaders=self.train_loader, ckpt_path=ckpt_path, val_dataloaders=self.test_loader)


def main(args, env_args):

    kstr = 'sim_action' if args.use_sim_action else 'vis_action'

    args.task_name = '1obj_seq2000_DexRep_pro100_start_uniform_{}_dsam_mod'.format(kstr)
    args.policy.actor_critic = 'ActorCriticDexRep'
    env_args.env.obs_dim.pop('pnG')

    args.seq_num=100
    args.add_noise=False
    args.noise_val=0.02

    args.exp_dir = os.path.join(args.exp_dir, args.task_name)
    os.makedirs(args.exp_dir,exist_ok=True)

    ds_train = GraspM3DexRepDataset(args, ds_name='train')
    ds_test = GraspM3DexRepDataset(args, ds_name='test')

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True) #**args.dataloader
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, drop_last=True,num_workers=4, pin_memory=True)

    trainer = BCTrainer(args,env_args, train_loader, test_loader)
    trainer.train()

if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load("{}/lhm_bc.yaml".format('../ActionDiffusion/bc/config'))
    env_args = OmegaConf.load("{}/shadow_hand_grasp_dexrep_ijrr.yaml".format('./cfg'))
    main(args, env_args)


