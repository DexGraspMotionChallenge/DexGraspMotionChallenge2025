import warnings

import gym
import torch
from gym.envs.registration import register
import numpy as np


def RegisterEnv(obj_name=None, env_py='grasp0_v0', env_maker=True, render=False, mode='template'):

    assert obj_name is not None, 'check for input obj_name, which is None'
    assert isinstance(env_py, str)
    if mode=='grab':
        mode='template'
    obj_name = obj_name[0].capitalize()+obj_name[1:]
    try:
        register(id=obj_name + '-v0',
                 entry_point='grasp_envs.grasp_suite.task.%s.%s:GraspEnvV0' % (mode, env_py),
                 max_episode_steps=200,
                 kwargs={'obj_name': obj_name})
    except:
        pass
    if env_maker:
        env = gym.make(obj_name + '-v0', render=render)
        return env
def get_T_matrix(Roation, xyz):
    if Roation.size == 3:
        matrix = R.from_euler('xyz', Roation.flatten(), degrees=False).as_matrix()
    elif Roation.size == 9:
        matrix = Roation
    else:
        raise AttributeError
    T = np.eye(4)
    T[:3, :3] = matrix
    T[:3, -1] = xyz

    return T

def pre_process_demos(demos, obj_names):
    '''collect obj demos we need'''

    demos_processed = []
    conmaps = dict()
    for demo in demos:
        if demo['obj_name'] in obj_names:
            for i, obs in enumerate(demo['observations']):
                demo['observations'][i] = np.concatenate([obs['hand_proprioception'], obs['object_represention']])
            demo['observations'] = np.vstack(demo['observations'])
            demos_processed.append(demo)
        # conmap = demo['con_map'].numpy()/55
        # conmap_idx = np.where(conmap>0.7)[0]
        # if demo['obj_name'] not in conmaps.keys():
        #     conmaps[demo['obj_name']] = [conmap_idx]
        # else:
        #     conmaps[demo['obj_name']].append(conmap_idx)
    # conmpa_save_path = '../demonstrations/ud_conmap.pkl'
    # if not os.path.exists(conmpa_save_path):
    #     pickle.dump(conmaps, open(conmpa_save_path, 'wb'))
    return demos_processed

def multi_envs_eval(policy, eval_rollouts, eval_objs):

    scores = np.zeros(len(eval_objs))
    for idx, obj_name in enumerate(eval_objs):
        env = GymEnv(obj_name.capitalize()+'-v0', render=False)
        score = env.evaluate_policy(policy, num_episodes=eval_rollouts, mean_action=True)
        print("%s:\tScore with behavior cloning = %f" % (obj_name, score[0][0]))
        scores[idx] = score[0][0]
    print('The average score of all obj is %f' % np.mean(scores))



def penetration(hand_m, obj_vox, pitch=0.005):
    # n_frames = len(traj['hand_g_orient'].shape[0])
    hand_mesh = hand_m
    obj_points = obj_vox.points
    with warnings.catch_warnings():

        inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    # volume = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)

    return volume * 1e6

def get_sample_intersect_volume(hand_mesh, obj_mesh, pitch=0.005):
    # hand_mesh = trimesh.Trimesh(vertices=sample_info["hand_verts_gen"], faces=hand_face)
    # obj_mesh = trimesh.Trimesh(vertices=sample_info["obj_verts_ori"], faces=sample_info["obj_faces_ori"])

    volume = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
    # volume = intersect_vox(hand_mesh, obj_mesh, pitch=0.005)
    return volume * 1e6

def intersect_vox(obj_mesh, hand_mesh, pitch=0.005):

    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume

def pc_normalize(pc, only_scale=False):
    """
    对点云数据进行归一化
    :param pc: 需要归一化的点云数据
    :param only_scale: keep transition info, only scale the size of the pcd
    :return: 归一化后的点云数据
    """

    # 求质心，也就是一个平移量，实际上就是求均值
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # 对点云进行缩放
    pc = pc / m
    if only_scale:
        pc = pc + centroid
    return pc
def pc_normalize_tensor(pc, only_scale=False):
    """
    对点云数据进行归一化
    :param pc: 需要归一化的点云数据
    :param only_scale: keep transition info, only scale the size of the pcd
    :return: 归一化后的点云数据
    """

    # 求质心，也就是一个平移量，实际上就是求均值
    centroid = torch.mean(pc, dim=-2)   # (N, 3)
    pc = pc - centroid.unsqueeze(-2)    # (N, 2048, 3) - (N, 1, 3) -> (N, 2048, 3)
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=-1)), dim=-1)[0]  # (N, )
    # 对点云进行缩放
    pc = pc / m.reshape(-1, 1, 1)   # (N, 2048, 3)
    if only_scale:
        pc = pc + centroid.unsqueeze(-2)
    return pc
def pc_normalize2(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # 对点云进行缩放
    pc = pc / m
    return pc, centroid, m
def pc_normalize2_tensor(pc):
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))

    # 对点云进行缩放
    pc = pc / m
    return pc, centroid, m
def denormalize(points, centroid, m):
    points = points * m
    points = points + centroid

    return points

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value

def split_batch_process(split_batch, input_tensor_list, process_chain):
    batch_num = input_tensor_list[0].shape[0]
    batch_split_num = int((batch_num + split_batch - 1) / split_batch)
    # split input
    split_input_list = []
    for it in input_tensor_list:
        assert batch_num == it.shape[0]
        it_slices = [
            it[i*split_batch:(i+1)*split_batch, ...] \
            for i in range(batch_split_num)
        ]
        split_input_list.append(it_slices)
    # transpose input list
    split_input_list = [list(item) for item in zip(*split_input_list)]
    # process
    result_list = []
    for iter in range(batch_split_num):
        chain_idx = 0
        result = None
        for my_func in process_chain:
            if chain_idx == 0: # input
                result = my_func(*split_input_list[iter])
            else:
                result = my_func(result)
            chain_idx += 1
        result_list.append(result)

def split_torch_dist(finger_points, obj_pcb, split_batch):
    assert finger_points.shape[0] == obj_pcb.shape[0]
    batch_num = finger_points.shape[0]
    batch_split_num = int((batch_num + split_batch - 1) / split_batch)
    # split input
    finger_points_split = [
        finger_points[i*split_batch:(i+1)*split_batch, ...] \
            for i in range(batch_split_num)
    ]
    obj_pcb_split = [
        obj_pcb[i * split_batch:(i + 1) * split_batch, ...] \
            for i in range(batch_split_num)
    ]
    # process
    dis_min_slices, dis_min_idx_slices = [], []
    for iter in range(batch_split_num):
        dis_split = torch.cdist(finger_points_split[iter], obj_pcb_split[iter])
        dis_min_split, dis_min_idx_split = torch.min(dis_split, dim=-1)
        dis_min_slices.append(dis_min_split)
        dis_min_idx_slices.append(dis_min_idx_split)
    # cat
    dis_min = torch.cat(dis_min_slices, dim=0)
    dis_min_idx = torch.cat(dis_min_idx_slices, dim=0)

    return dis_min, dis_min_idx