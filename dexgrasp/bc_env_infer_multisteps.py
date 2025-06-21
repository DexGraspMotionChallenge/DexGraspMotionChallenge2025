import gc
import os
import os.path as osp
import numpy as np
import json
from glob import glob
from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_marl import get_AgentIndex

from ActionDiffusion.bc.model.policy.lhm_policy import LitDP3Model
from utils.test_env import test_env
from utils.gym_util.multistep_wrapper import MultiStepWrapper
from utils.info_summary_print import save_results_summary

import torch
import json

def create_env(bc_cfg=None):

    if args.num_objs != -1:
        cfg['env']['num_objs'] = args.num_objs
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    cfg['env']['env_mode'] = 'bc_env_infer'

    if bc_cfg.obs_type=='dexrep':
        cfg["env"]["observationType"]='DexRep'
    if bc_cfg.obs_type=='pcds':
        cfg["env"]["observationType"]='obs_pcds'

    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
    env = MultiStepWrapper(env, bc_cfg.n_obs_steps, bc_cfg.n_action_steps, max_episode_steps=200)

    return task, env

def create_bc_model(base_path):
    from omegaconf import OmegaConf
    import os.path as osp

    bc_args = OmegaConf.load("{}/dp3.yaml".format(osp.join(base_path, 'ActionDiffusion/bc/config')))
    env_args = OmegaConf.load("{}/shadow_hand_grasp_dexrep_ijrr.yaml".format(osp.join(base_path, 'dexgrasp/cfg')))
    bc_model_name = bc_args.policy.actor_critic
    if bc_model_name=='ActorCriticPNG':
        env_args.env.obs_dim.pop('dexrep_sensor')
        env_args.env.obs_dim.pop('dexrep_pnl')

    elif bc_model_name=='ActorCriticDexRep':
        env_args.env.obs_dim.pop('pnG')
        bc_args.policy.dexrep_encoder_cfg.env_cfg.obs_dim.pop('pnG')

    if bc_args.obs_type=='dexrep':
        bc_args.obs_as_global_cond=False
        bc_args.obs_as_dexrep_cond=True

    if bc_args.obs_type=='pcds':
        bc_args.obs_as_global_cond=True
        bc_args.obs_as_dexrep_cond=False

    bc_model = LitDP3Model(bc_args, env_args.env)
    bc_model = bc_model.to(args.rl_device)
    bc_args.policy.checkpoints = osp.join(base_path, bc_args.policy.checkpoints.split('DexRep_Isaac_ijrr/')[-1])
    ckpt = torch.load(bc_args.policy.checkpoints, map_location=torch.device(args.rl_device))
    bc_model.load_state_dict(ckpt['state_dict'])

    snapshot_names = os.path.basename(bc_args.policy.checkpoints).split('.')[0]+'_'+bc_args.policy.checkpoints.split('/')[-2]
    bc_info_name= cfg['env']['obj_type']+'_'+snapshot_names #+'_'bc_model_name
    return bc_model, bc_model_name, bc_info_name, bc_args

def run(mode='seen'):
    base_path = '../'
    if mode=='seen' or 'one':
        folder = cfg['trajs_path']['train'].split('/')[-1]
        obj_id_list =cfg['env']['seen_object_code_dict']
    else:
        folder = cfg['trajs_path']['valid'].split('/')[-1]
        obj_id_list =cfg['env']['unseen_object_code_dict']
    data_path = os.path.join(base_path, 'dexgrasp/{}'.format(folder))

    bc_model, bc_model_name, bc_info_name, bc_args = create_bc_model(base_path)
    bc_info_name=bc_info_name+'_'+'test_num{}'.format(cfg['env']['test_num'])
    cfg['env']['bc_model_name'] = bc_model_name

    results = {'total_succ_rates':[],'dataset_name':cfg['trajs_path']['train'],'detail':[]}
    print('bc info: {}'.format(bc_info_name))
    batch_size = cfg['env']['infer_batch_size']
    num_full_batches = len(obj_id_list) // batch_size
    for i in range(0, num_full_batches * batch_size, batch_size):
        batch = obj_id_list[i:i + batch_size]
        processed_batch = [obj_id[:-4] if obj_id.endswith('.npy') else obj_id for obj_id in batch]
        cfg['env']['object_code_dict'] = processed_batch
        obj_glob_feat = None
        task, env = create_env(bc_args)

        succ_rate, result_desc = test_env(args, task, env, bc_model, bc_model_name, processed_batch[-1], obj_glob_feat)
        results['total_succ_rates'].append(succ_rate)
        results['detail'].append([result_desc])
        env.task.clean_sim()
        del task,env
        gc.collect()
    save_results_summary(results, filename=bc_info_name, to_yaml=True)

    print('---------------------finish {}--------------------------\n'.format(bc_info_name))


if __name__ == '__main__':
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    agent_index = get_AgentIndex(cfg)

    if cfg['env']['obj_type'] == 'seen':
        info = np.load('./train_file_names.npy', allow_pickle=True).item()
        cfg['env']['seen_object_code_dict'] = info['train_obj_num_1h']
    elif cfg['env']['obj_type'] == 'one':
        cfg['env']['seen_object_code_dict'] = ['core-bottle-a02a1255256fbe01c9292f26f73f6538']
        cfg['env']['infer_batch_size'] = 1

    run(mode=cfg['env']['obj_type'])
