import torch
import statistics
import torch.nn.functional as F
from utils.torch_jit_utils import *


# remove(56,84), (149-179), [84-149]->(5,13)->cut->(5,7)->(1,35), (216-222) : 222-28-30-30-6=94
def obs_process(observation,pro_dim=128):
    """
    observation =(N,2582) or (n_obs_steps, N, 2582)
    """

    reshape84_149_part =torch.arange(84,149).reshape(5,13) #
    remove_indice_part = reshape84_149_part[:,-6:].reshape(-1) #fingertips pos&ang vel 30

    remmove28_56 = torch.arange(28,56) # shadow vel 28
    remove56_84 = torch.arange(56,84) # shadow force 28

    remove149_179 = torch.arange(149, 179) #finger force 30
    remove216_222 = torch.arange(216, 222) #obj pos&ang vel 6

    index_to_remove = torch.cat([remove56_84,remove149_179])
    if pro_dim==134:
        final_indice_remove = torch.cat([index_to_remove, remove_indice_part])
    elif pro_dim==128:
        final_indice_remove = torch.cat([index_to_remove, remove_indice_part, remove216_222])
    elif pro_dim==100:
        final_indice_remove = torch.cat([index_to_remove, remove_indice_part, remove216_222, remmove28_56])


    all_indices = torch.arange(2582)

    mask = ~torch.isin(all_indices, final_indice_remove)
    filter_observation = observation[...,mask]
    return filter_observation

@torch.no_grad()
def test_env(args, task, env, model, bc_model_name,obj_id, global_feat=None,use_gt=False, use_part_gt=False):
    if use_gt or use_part_gt:
        gt_actions = torch.tensor(task.obj_trajs_info['grasp_seqs']).to('cuda')
    # traj_obs = torch.tensor(task.obj_trajs_info['obs']).to('cuda')

    model.eval()
    pro_dim = task.cfg['env']['obs_dim']['prop']
    maxlen = 50000
    cur_reward_sum = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    cur_episode_length = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    cur_success = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)
    cur_success_done_sum = torch.zeros(task.num_envs, dtype=torch.float, device=args.rl_device)

    task.reset_buf = torch.ones(task.num_envs, device=task.device, dtype=torch.long)
    task.progress_buf = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)
    current_obs_state = env.reset()
    # current_obs_state= traj_obs[:,0]
    if current_obs_state.shape[-1]==2582:
        current_obs_state = obs_process(current_obs_state,pro_dim=pro_dim) #2582->2488 if LitDP3Model: (2,N, 2488); LitBCModel: (N, 2488)
    if len(current_obs_state.size())==3 or len(current_obs_state.size())==4:
        current_obs_state = current_obs_state.transpose(0,1) #(N, 2, 2488)
    a=1
    if bc_model_name=='ActorCriticPNG':
        current_obs_state = torch.cat([current_obs_state[:,:pro_dim], global_feat.repeat(task.num_envs,1)], dim=-1)
        a=1

    reward_sum = []
    episode_length = []

    success_sum_list = []
    success_done_list = []

    i=0
    if model.__class__.__name__ == 'LitDP3Model':
        if model.args.use_orig_encoder:
            unactions = torch.tensor(task.obj_trajs_info['vis_unscale_actions'])[:,0:1,:]\
                .repeat(1,model.args.n_obs_steps,1).to(current_obs_state.device)#(N,2,28)
            a=1


    while len(reward_sum) < maxlen:
        # print(f'Frame {i+1} ---------------')
        # ep_infos = []
        # actions, state, obs_feat = actor.act_inference(current_obs_state)

        if model.__class__.__name__ == 'LitDP3Model':
            input_dict = {'obs': current_obs_state}
            if model.args.use_orig_encoder:
                input_dict['actions']=unactions
            actions_dict = model.model.predict_action(input_dict) #obs: (N,2, 2488) actions: (N,3, 28)
            actions = actions_dict['natcion'].squeeze(0).transpose(0,1) #actions: (3,N, 28)naction_pred

        elif model.__class__.__name__ == 'LitBCModel':
            actions = model.model.act_inference(current_obs_state)
            a=1
        actions = actions.clamp(-1,1)
        if use_gt or use_part_gt:
            if use_part_gt and i>5:
                pass
            elif len(current_obs_state.size())==3:
                actions_gt = unscale(gt_actions[:, i*3:i*3+3], task.shadow_hand_dof_lower_limits, task.shadow_hand_dof_upper_limits)
                actions = actions_gt.transpose(0,1)
            else:
                actions_gt = unscale(gt_actions[:, i], task.shadow_hand_dof_lower_limits, task.shadow_hand_dof_upper_limits)
                actions = actions_gt
            a=1

        # actions = torch.zeros_like(actions).to('cuda')

        # Step the vec_environment
        # next_obs, rews, dones, infos = vec_env.step(actions)

        if model.__class__.__name__ == 'LitDP3Model':
            nsteps = len(actions)
            obs_buf,_,_,_ = env.step(actions, i*nsteps) #(2, N, 2488)
            next_obs = obs_buf.transpose(0,1).clone() #(N, 2, 2488)
        elif model.__class__.__name__ == 'LitBCModel':
            if i>30 and env.task.cfg['env']['use_pre_fixed_actions']: #40
                actions = env.task.get_pre_target_actions(actions,i)
            env.task.step(actions, i+1)
            next_obs = env.task.obs_buf.clone()# (B,2580)
            # next_obs = traj_obs[:, i]

        # next_obs_gt = obs_process(traj_obs[:,i+1],pro_dim=pro_dim)
        # next_obs[:,185:209] = scale(next_obs[:,185:209], task.shadow_hand_dof_lower_limits[:24], task.shadow_hand_dof_upper_limits[:24])
        if next_obs.shape[-1] == 2582:
            next_obs = obs_process(next_obs,pro_dim=pro_dim) #2582->2488
        # next_obs_gt = obs_process(next_obs_gt,pro_dim=pro_dim)
        a=1

        if bc_model_name == 'ActorCriticPNG':
            next_obs = torch.cat([next_obs[:,:pro_dim], global_feat.repeat(task.num_envs,1)], dim=-1)
        rews = env.task.rew_buf.clone() #(B, )
        dones = env.task.reset_buf.clone() #(B, )

        # next_states = vec_env.get_state()
        # Record the transition
        # storage.add_transitions(state, obs_feat, actions_expert, rews, dones)
        current_obs_state.copy_(next_obs) #(B, 2582)

        if model.__class__.__name__ == 'LitDP3Model':
            if model.args.use_orig_encoder:
                unactions.copy_(actions.transpose(0,1)[:,-model.args.n_obs_steps:,:]) #(N,n_obs_steps,28)
                # unactions.copy_(task.unactions.unsqueeze(1).repeat(1,model.args.n_obs_steps,1)) #(N,n_obs_steps,28)
                a=1

        # current_obs_state = traj_obs[:,i+1]
        # current_obs_state = obs_process(current_obs_state,pro_dim=pro_dim)  # 2582->2488

        # Book keeping
        # ep_infos.append(infos)
        cur_reward_sum[:] += rews
        cur_episode_length[:] += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)
        success_ = torch.where(task.successes==1)[0]*1 #(B, )

        cur_success[:] = task.successes #torch.where(task.successes==1)[0]*1
        cur_success_done_sum[new_ids] = cur_success[new_ids]
        if task.successes.sum()>0:
            print('name={}, iter={}, N_seq={}, success_num:{}'.format(obj_id, i ,task.num_envs, task.successes.sum()))
            a=1
        else:
            print('name={}, iter={}, N_seq={}, None'.format(obj_id, i ,task.num_envs))
            a=1
        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        # successes.extend(infos['successes'][new_ids.cpu()][:, 0].cpu().numpy().tolist())

        success_sum_list.append(cur_success.cpu().numpy().sum())
        success_done_list.append(cur_success[new_ids].cpu().numpy())

        # if len(new_ids) > 0:
        if (i>15 and model.__class__.__name__ == 'LitDP3Model') or (i>120 and model.__class__.__name__ == 'LitBCModel'):
            print('-' * 20)
            # donw_succ = torch.zeros(task.num_envs)
            # succ_rate = statistics.mean(cur_success.cpu().numpy())
            # donw_succ_rate = cur_success_done_sum.sum()/len(new_ids)
            succ_sum = max(success_sum_list)
            succ_rate = succ_sum/task.num_envs

            print(f'Mean Success {obj_id}::> {succ_rate.item() * 100:.2f}')

            result_desc = 'name={}, N_seq={}, success_num:{},success_rate:{}'.format(obj_id,task.num_envs, succ_sum, succ_rate)
            print(result_desc)
            break
            # print(f'Done Mean Success {obj_id}::> {donw_succ_rate * 100:.2f}')
        i+=1

        cur_reward_sum[new_ids] = 0
        cur_episode_length[new_ids] = 0
        cur_success_done_sum[new_ids] = 0
        if i>300:
            break


    return succ_rate, result_desc
    # return  obj_id, task.num_envs, task.successes.mean().cpu().numpy().item(),task.successes.sum().cpu().numpy().item()


