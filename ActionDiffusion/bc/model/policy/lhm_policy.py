import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from ActionDiffusion.bc.model.policy.lqt_policy import ActorCriticDexRep,ActorCriticPNG
from  ActionDiffusion.model.policy.dp3_lightning import DP3Lightning
from ActionDiffusion.utils.parser_util import simple_instantiate
from ActionDiffusion.common.pytorch_util import dict_apply
from torch.distributions import MultivariateNormal
import numpy as np

import copy

# import pathlib
# import hydra


class LitBCModel(pl.LightningModule):
    def __init__(self, args,env_args):
        super().__init__()
        self.args = args

        self.model_cfg = self.args.policy
        self.actions_shape =  self.model_cfg.actions_shape
        self.learn_cfg = self.args.learn
        self.init_noise_std = self.learn_cfg.get("init_noise_std", 0.3)
        self.encoder_cfg = self.args.encoder
        self.cfg_env = env_args
        self.model = eval(self.model_cfg.actor_critic)(None, self.actions_shape, self.init_noise_std,self.model_cfg, self.encoder_cfg, self.cfg_env)

        a=1
    def forward(self, batch):
        actions_mean = self.model(batch['obs'])
        return actions_mean


    def training_step(self, batch, batch_idx):
        pred_action = self.forward(batch) #(B, 3+6+22)
        loss_dct = self.cal_loss(pred_action,batch['actions'])

        self.log_dict({'train_loss':loss_dct['loss']}, prog_bar=True, on_epoch=True)
        self.log_dict({'wrist_loss':loss_dct['wrist_loss']}, prog_bar=True, on_epoch=True)
        self.log_dict({'ori_loss':loss_dct['ori_loss']}, prog_bar=True, on_epoch=True)
        self.log_dict({'finger_loss':loss_dct['finger_loss']}, prog_bar=True, on_epoch=True)
        self.log_dict({'l1_loss':loss_dct['l1_loss']}, prog_bar=True, on_epoch=True)

        return loss_dct['loss']

    def validation_step(self, batch, batch_idx):
        pred_action = self.forward(batch) #(B, 3+6+22)
        loss_dct = self.cal_loss(pred_action, batch['actions'])

        self.log_dict({'val_loss':loss_dct['loss']}, prog_bar=True, on_epoch=True)
        self.log_dict({'val_wrist_loss':loss_dct['wrist_loss']}, prog_bar=True, on_epoch=True)
        self.log_dict({'val_ori_loss':loss_dct['ori_loss']}, prog_bar=True, on_epoch=True)
        self.log_dict({'val_finger_loss':loss_dct['finger_loss']}, prog_bar=True, on_epoch=True)
        self.log_dict({'val_l1_loss':loss_dct['l1_loss']}, prog_bar=True, on_epoch=True)

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
    #
    #     self.log('avg_val_loss', avg_loss)
    #     self.log('avg_val_acc', avg_acc)


    def cal_loss(self, predict, gt):
        wrist_loss = F.mse_loss(predict[:, :3], gt[:, :3])
        ori_loss = F.mse_loss(predict[:, 3:6], gt[:, 3:6])
        finger_loss = F.mse_loss(predict[:, 6:], gt[:, 6:])
        l1_loss = F.l1_loss(predict, gt)

        loss = 2 * wrist_loss + ori_loss + finger_loss+ l1_loss

        # loss = F.mse_loss(predict, gt)
        # if self.current_epoch in [1, 5,10,20]:
        #     a=1

        loss_dict  ={
            "loss": loss,
            'wrist_loss': wrist_loss,
            'ori_loss': ori_loss,
            'finger_loss':finger_loss,
            'l1_loss': l1_loss,
        }

        return loss_dict

    def configure_optimizers(self):
        params_list = [{'params': self.parameters(), 'lr': self.args.lr}]
        optimizer = torch.optim.Adam(params_list)
        return {"optimizer": optimizer}


class LitDP3Model(pl.LightningModule):
    def __init__(self, args, env_args):
        super().__init__()
        self.args = args
        self.model_cfg = self.args.policy
        self.actions_shape = self.model_cfg.actions_shape
        # self.learn_cfg = self.args.learn
        # self.init_noise_std = self.learn_cfg.get("init_noise_std", 0.3)
        # self.encoder_cfg = self.args.encoder
        # self.cfg_env = env_args
        self.model = simple_instantiate(args.policy) # DP3Lightning
        # self.log_std = nn.Parameter(np.log(0.8) * torch.ones(*self.model.action_shape))

        self.ema_model=None
        if args.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:  # minkowski engine could not be copied. recreate it
                self.ema_model = simple_instantiate(args.policy)
        self.log_std = self.model.log_std

    def forward(self, batch):
        pred, target, loss_mask = self.model(batch)

        return pred, target, loss_mask


    def training_step(self, batch, batch_idx):
        pred, target, loss_mask = self.forward(batch) #(B, 3+6+22)
        loss, loss_dict = self.model.compute_loss(pred, target, loss_mask)

        for k,v in loss_dict.items():
            self.log_dict({'train_'+k: v}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pred, target, loss_mask = self.forward(batch) #(B, 3+6+22)
        loss, loss_dict = self.model.compute_loss(pred, target, loss_mask)

        for k,v in loss_dict.items():
            self.log_dict({'val_'+k: v}, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        params_list = [{'params': self.parameters(), 'lr': self.args.lr}]
        optimizer = torch.optim.Adam(params_list)
        return {"optimizer": optimizer}

    def act(self, observations):
        """
        current_obs_state (B,N,2460)
        """
        local_cond = None
        global_cond = None

        if observations.size()[-1] == 2582:
            observations = observations[..., self.model.obs_encoder.dexrep_encoder.obs_mask]

        T = self.model.horizon #(4)
        Da = self.model.action_dim
        Do = self.model.obs_feature_dim
        To = self.model.n_obs_steps
        dtype = self.model.dtype

        batch = {'obs':observations}

        if self.model.obs_as_dexrep_cond:
            B, N, D = observations.size()
            obs_dim = self.args.policy.dexrep_encoder_cfg.env_cfg.obs_dim.prop
            state =observations[...,:obs_dim]

            this_nobs = dict_apply(batch, lambda x: x[:, :self.model.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
                                   skip_key=['obj_pcd', 'actions']) #(B*n_obs_steps, D)

            nobs_features = self.model.obs_encoder(this_nobs['obs']) #(B*n_obs_steps, ...)
            global_cond = nobs_features.reshape(B, -1)  #(B, n_obs_steps*D)

            cond_data = torch.zeros(size=(B, T, Da), device=self.device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            a=1

        # run sampling
        nsample = self.model.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.model.kwargs)

        # unnormalize prediction
        action_pred_mean = nsample[..., :Da]
        start = To - 1
        end = start + self.model.n_action_steps
        actions_mean = action_pred_mean[:,start:end] #(B, 7, 28)
        # actions_mean_flat = actions_mean.reshape(B, -1)  # (B, 28*7)
        actions_mean_flat = actions_mean.reshape(-1, Da)  # (B*T, 28)

        covariance = torch.diag(self.model.log_std.exp() * self.model.log_std.exp())
        # distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        distribution = MultivariateNormal(actions_mean_flat, scale_tril=covariance)

        actions = distribution.sample() #(B*T,28) or (B,T*28)

        # actions_log_prob = distribution.log_prob
        actions_log_prob = distribution.log_prob(actions).reshape(B,-1).mean(dim=-1) #(B,T)

        actions = actions.reshape(B, -1, Da)

        # nobs_features = nobs_features.reshape(B, -1, 384).mean(dim=1)
        value = self.model.obs_encoder.dexrep_encoder.critic(global_cond) #(B,384*n_obs)

        # sigma = self.model.log_std.repeat(actions_mean.shape[0], 1).detach()
        sigma = self.model.log_std.repeat(actions_mean.shape[0],(end-start), 1).detach()

        # sigma = self.model.log_std.repeat(actions_mean.shape[0], 1).reshape(B,-1,Da).detach()

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               sigma, \
               state.detach(),\
               observations[..., state.shape[2]:].detach()

    def evaluate(self, obs_features, state, actions):
        B, _, D = obs_features.size()

        T = self.model.horizon #(4)
        Da = self.model.action_dim
        To = self.model.n_obs_steps

        dtype = self.model.dtype

        local_cond = None
        # global_cond = None

        nobs_features = self.model.obs_encoder.dexrep_encoder.evaluate_(obs_features, state)  # (N,384)


        global_cond = nobs_features.reshape(B, -1)  # (B, n_obs_steps*D)

        cond_data = torch.zeros(size=(B, T, Da), device=self.device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.model.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.model.kwargs)
        action_pred_mean = nsample[..., :Da] #(B,8,28)
        # get action
        start = To - 1
        end = start + self.model.n_action_steps

        # action = action_pred[:, start:end]
        actions_mean = action_pred_mean[:,start:end] #(B,7,28)
        actions_mean_flat = actions_mean.reshape(-1, Da)  # (B*T, 28)

        covariance = torch.diag(self.model.log_std.exp() * self.model.log_std.exp())
        distribution = MultivariateNormal(actions_mean_flat, scale_tril=covariance)
        entropy = distribution.entropy()

        # actions = distribution.sample() #(B*T,28) or (B,T*28)
        actions_log_prob = distribution.log_prob(actions.reshape(-1,Da)).reshape(B,-1).mean(dim=-1) #(B,T)
        # actions = actions.reshape(B, -1, Da)

        # nobs_features = nobs_features.reshape(B, -1, 384).mean(dim=1)
        value = self.model.obs_encoder.dexrep_encoder.critic(global_cond) #(B,1)


        sigma = self.model.log_std.repeat(actions_mean.shape[0],(end-start), 1).detach()


        return actions_log_prob, entropy, value, actions_mean, sigma

# def main(cfg):
#     import torch
#     batch_size = 4
#     cfg.policy.batch_size = batch_size
#     model = LitBCModel(cfg.policy)
#
#
#     batch = {'pcd': torch.rand(batch_size,2048,3),
#              'start_state':torch.rand(batch_size,9),
#               'h2o_vec_state':torch.rand(batch_size,19, 3),
#              }
#     out =model.bc_forward(batch)
#     a=1
#
#
# if __name__ == '__main__':
#     main()