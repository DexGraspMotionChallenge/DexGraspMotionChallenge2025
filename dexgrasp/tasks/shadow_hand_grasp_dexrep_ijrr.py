# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os
import os.path as osp
import numpy as np
import torch
import trimesh
import random
from glob import glob
from tqdm import tqdm
from utils.torch_jit_utils import *
from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from dexrep.ShareDexRepSensor import SharedDexRepSensor as DexRepEncoder
from dexrep.ShareDexRepSensor import SharedPnGSensor as PnGEncoder
from scipy.spatial.transform import Rotation as R
import open3d as o3d
# from dexgrasp.utils.update_params import update_seq_grasp_direction_batch
from utils.update_params import update_seq_grasp_direction_batch

from pytorch3d.transforms import quaternion_to_matrix,euler_angles_to_matrix
from utils.hand_model import HandModel
from ActionDiffusion.utils.vis_utils import html_antmation_save
from dexgrasp.utils.traj_utils import modify_hand_trajectory,downsampling_trajectory,compute_h2o_minimum_vec,\
    rotate_trajs_and_object_to_zneg_vectorized, unwrap_euler_batch_vectorized

_DexRepEncoder_Map = {
            'pnG': PnGEncoder,
            'DexRep': DexRepEncoder,
            'DexRep_debug': DexRepEncoder,
        }

class ShadowHandGraspDexRepIjrr(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, npy_list=None):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)

        self.tips_idxs = [8,12,16,21]
        self.big_tips_idx = [27]
        self.per_obj_seq_idx=None
        self.object_idxs = [0]
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        model_base_path = "../assets/mjcf/"
        self.hand_model = HandModel(
            mjcf_path=model_base_path + 'shadow_hand_vis_new.xml', mesh_path=model_base_path + 'meshes',
            contact_points_path=model_base_path + 'contact_points.json',
            penetration_points_path=model_base_path + 'penetration_points.json',
            n_surface_points=512,
            device='cpu',#device_type #"cpu"
            use_joint21=True
        )
        a=1



        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)
        self.obs_type = self.cfg["env"]["observationType"]
        print("Obs type:", self.obs_type)

        num_obs = 236 + 64
        self.num_obs_dict = {
            "full_state": num_obs,
            "DexRep": 2567,
            "obs_pcds": (1024, 3)
        }
        # if use DexRep Encoder
        if self.obs_type in _DexRepEncoder_Map.keys():
            assert "dexrep" in cfg.keys()
            self.use_dexrep = True
            self.use_pnG = False

            self.DexRepEncoder = _DexRepEncoder_Map[self.obs_type](cfg, device_type + f":{device_id}")

        elif self.obs_type == 'obs_pcds' or self.obs_type == 'obs_h2o':
            self.use_dexrep = False
            self.use_pnG = False
            self.use_geodex = False
            self.use_obs_pcds = True
        else:
            self.use_dexrep = False

        if self.cfg['env']['bc_model_name'] == 'ActorCriticPNG':
            self.use_dexrep = False
            self.use_pnG = True

        self.num_hand_obs = 66 + 95 + 24 + 6  # 191 =  22*3 + (65+30) + 24
        self.up_axis = 'z'
        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal",
                           "robot0:thdistal"]
        self.hand_center = ["robot0:palm"]
        self.num_fingertips = len(self.fingertips)
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        num_states = 0
        if self.asymmetric_obs:
            num_states = 211
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.num_agents = 1
        self.cfg["env"]["numActions"] = 28
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.dexrep_hand = [
            "robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal",
            "robot0:ffmiddle", "robot0:mfmiddle", "robot0:rfmiddle", "robot0:lfmiddle", "robot0:thmiddle",
            "robot0:ffproximal", "robot0:mfproximal", "robot0:rfproximal", "robot0:lfmetacarpal", "robot0:thproximal"
        ]

        if cfg['env']['env_mode'] in ['bc_env_infer', 'rl_learning'] :
            trajs_path = "./dataset/train"

        if cfg['env']['env_mode']=='rl_learning':
            self.obj_trajs_info = self.rl_data_load(trajs_path)

        else:
            npy_list = []
            for obj_id in self.cfg['env']['object_code_dict']:
                npy_path = os.path.join(trajs_path, f"{obj_id}.npy")
                data = np.load(npy_path, allow_pickle=True).item()
                npy_list.append(data)
            self.batch_load_data_dict(npy_list)

        self.obj_trajs_info['grasp_seqs'] = unwrap_euler_batch_vectorized( self.obj_trajs_info['grasp_seqs'])

        self.cfg['env']['numEnvs'] = self.obj_trajs_info['grasp_seqs'].shape[0]
        self.table_height = self.cfg['env']['table_height']

        if self.cfg['env']['traj_modify']:
            self.obj_trajs_info['grasp_seqs'][:,:40,:] = modify_hand_trajectory(self.obj_trajs_info['grasp_seqs'][:,:40,:])
            a=1

        if  self.cfg['env']['traj_down_sample']:
            self.obj_trajs_info['grasp_seqs'] = downsampling_trajectory( self.obj_trajs_info['grasp_seqs'])

        self.Rz = None

        if cfg['env']['seq_start_pos_uniform'] and cfg['env']['env_mode']=='extract_obs':
            grasp_seqs = torch.from_numpy(self.obj_trajs_info['grasp_seqs'])

            if cfg['env']['seq_start_rot_uniform']:
                grasp_seqs, R_align = rotate_trajs_and_object_to_zneg_vectorized(grasp_seqs)
                obj_rotmat = self.obj_trajs_info['obj_rotmat']  # (N, 3, 3)
                self.obj_trajs_info['obj_rotmat'] = np.matmul(R_align.numpy(), obj_rotmat)  # (N, 3, 3)

            grasp_seqs, Rz = update_seq_grasp_direction_batch(grasp_seqs) #(N,T, 28) (N, 3, 3)

            self.obj_trajs_info['grasp_seqs'] = grasp_seqs.numpy()
            self.grasp_seqs = grasp_seqs.to(torch.device(device_type + f":{device_id}"))

            obj_rotmat = self.obj_trajs_info['obj_rotmat'] #(N, 3, 3)
            self.obj_trajs_info['obj_rotmat'] = np.matmul(Rz.numpy(), obj_rotmat) #(N, 3, 3)
        else:
            self.grasp_seqs = torch.from_numpy(self.obj_trajs_info['grasp_seqs']).to(
                torch.device(device_type + f":{device_id}"))

        super().__init__(cfg=self.cfg, enable_camera_sensors=True)

        self.num_dexrep_hand = len(self.dexrep_hand)
        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # if self.obs_type == "full_state" or self.asymmetric_obs:
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
                                                self.num_shadow_hand_dofs + self.num_object_dofs)
        self.dof_force_tensor = self.dof_force_tensor[:, :self.num_shadow_hand_dofs]


        self.sim_refresh()


        self.z_theta = torch.zeros(self.num_envs, device=self.device)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.total_successes = 0
        self.total_resets = 0

        self.pre_target_actions = torch.zeros((self.num_envs, 28), device=self.device, dtype=torch.float)
        self.check_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
        self.iter_check_store = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)

    def batch_load_data_dict(self, npy_list):
        # self.object_code_list = []
        self.object_idxs = []
        self.obj_trajs_info = {'obj_scale':[], 'obj_rotmat':[],'grasp_seqs':[],'vis_unscale_actions':[]}

        for obj_id, data_dct in enumerate(npy_list):
            # obj_code = data_dct['obj_code']
            # self.object_code_list.append(obj_code)

            obj_trajs_info, object_idxs = self.load_data_dict(data_dct, [obj_id])
            self.object_idxs+=object_idxs
            # for key, val in obj_trajs_info.items():
            #     self.obj_trajs_info[key].append(val)
            for key in self.obj_trajs_info:
                # if key in obj_trajs_info:
                self.obj_trajs_info[key].append(obj_trajs_info[key])

        for key, val in self.obj_trajs_info.items():
            self.obj_trajs_info[key] = np.concatenate(val,axis=0)
        a=1

    def load_data_dict(self, data_dict, obj_id=None):
        # data_dict.pop('obj_code')
        obj_trajs_info = data_dict

        if isinstance(obj_id,list):
            object_idxs = obj_id*obj_trajs_info['grasp_seqs'].shape[0]
        else:
            object_idxs = [0]* obj_trajs_info['grasp_seqs'].shape[0]

        return obj_trajs_info, object_idxs

    def rl_data_load(self, trajs_path):
        data_npt_list = glob(osp.join(trajs_path, "**.npy"))
        data_npt_list = sorted( [f for f in data_npt_list if os.path.getsize(f) >= 1024],key=lambda path: os.path.getsize(path),reverse=True)
        data_npt_list = data_npt_list[17:18]
        traj_data_dct = {'grasp_seqs':[], 'obj_rotmat':[], 'obj_scale':[]}

        self.cfg['env']['object_code_dict'] = []
        self.per_obj_seq_idx = {}
        self.object_idxs = []
        seq_idx=0
        for obj_id, path in enumerate(tqdm(data_npt_list, desc="Processing")):
            obj_code_name = osp.basename(path).split('.')[0]

            np_data_dct = np.load(path, allow_pickle=True).item()
            # obj_codes = [obj_code_name]*len(np_data_dct['grasp_seqs'])
            # self.cfg['env']['object_code_dict']+=obj_codes

            self.cfg['env']['object_code_dict'].append(obj_code_name)

            seq_num = len(np_data_dct['grasp_seqs'])
            self.per_obj_seq_idx[obj_id]=[seq_idx,seq_idx+seq_num]
            self.object_idxs+=[obj_id]*len(np_data_dct['grasp_seqs'])
            seq_idx +=seq_num
            a=1
            for key in traj_data_dct.keys():
                val = np_data_dct[key]
                traj_data_dct[key].append(val)


        for  key, val in traj_data_dct.items():
            traj_data_dct[key] = np.concatenate(val)


        print('RL Data Load Finish ObjNum={} -- SeqNum={}'.format(len(self.cfg['env']['object_code_dict']), seq_idx))
        return traj_data_dct


    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):

        object_code_list = self.cfg['env']['object_code_dict']
        self.object_code_list = object_code_list
        # all_scales = set()

        self.repose_z = self.cfg['env']['repose_z']

        self.grasp_data = {}
        assets_path = '../assets'
        print(f'Num Objs: {len(self.object_code_list)}')
        print(f'Num Envs: {self.num_envs}')

        self.goal_cond = self.cfg["env"]["goal_cond"]
        self.random_prior = self.cfg['env']['random_prior']
        self.random_time = self.cfg["env"]["random_time"]
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        shadow_hand_asset, shadow_hand_dof_props, table_texture_handle = self._load_shadow_hand_asset()

        goal_asset_dict, object_asset_dict = self._load_object_asset(assets_path)

        # create table asset
        table_asset, table_dims = self._load_table_asset()

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 1)  # gymapi.Vec3(0.1, 0.1, 0.65)
        self.init_hand_pos_z = 1.0
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0)  # gymapi.Vec3(0.0, 0.0, 0.72)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.3+self.table_height)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)

        goal_start_pose.p.z -= 0.0

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # compute aggregate size
        # max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * self.num_object_bodies + 1  ##
        # max_agg_shapes = self.num_shadow_hand_shapes * 1 + 2 * self.num_object_shapes + 1  ##

        self.shadow_hands = []
        self.objects = []
        self.envs = []
        self.object_init_state = []
        self.goal_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []
        self.dexrep_hand_indices = []
        for o in range(len(self.dexrep_hand)):
            dexrep_hand_env_handle = self.gym.find_asset_rigid_body_index(shadow_hand_asset, self.dexrep_hand[o])
            self.dexrep_hand_indices.append(dexrep_hand_env_handle)
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        body_names = {
            'wrist': 'robot0:wrist',
            'palm': 'robot0:palm',
            'thumb': 'robot0:thdistal',
            'index': 'robot0:ffdistal',
            'middle': 'robot0:mfdistal',
            'ring': 'robot0:rfdistal',
            'little': 'robot0:lfdistal'
        }
        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(shadow_hand_asset, body_name)

        # create fingertip force sensors, if needed
        # if self.obs_type == "full_state" or self.asymmetric_obs:
        sensor_pose = gymapi.Transform()
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)

        # self.object_scale_buf = {}
        self.obj_half_height_list = []

        self.obj_mesh_list = []
        self.obj_sample_points_list = []
        for obj_code in self.object_code_list:
            # dexrep_load = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj_raw"] + obj_code + "/coacd" + f'/decomposed.obj'
            dexrep_load = '../assets/meshdata/' + obj_code + "/coacd" + f'/decomposed.obj'
            obj_mesh = trimesh.load_mesh(dexrep_load)
            if isinstance(obj_mesh, trimesh.Scene):
                obj_mesh = trimesh.util.concatenate([geometry for geometry in obj_mesh.geometry.values()])
            obj_sample_points = self.get_object_sample_points(obj_mesh)
            self.obj_mesh_list.append(obj_mesh)
            self.obj_sample_points_list.append(obj_sample_points)


        if len(self.object_code_list)==1:
            # dexrep_load = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj_raw"] + self.object_code_list[0] + "/coacd" + f'/decomposed.obj'
            dexrep_load = '../assets/meshdata/' + obj_code + "/coacd" + f'/decomposed.obj'
            obj_mesh = trimesh.load_mesh(dexrep_load)
            self.obj_mesh = obj_mesh
            self.obj_sample_points = self.get_object_sample_points(obj_mesh)

        down_sample = True if self.obs_type == 'obs_pcds' else False
        self.obj_init_obj_pcds = self.get_batch_obj_pcds(down_sample=down_sample).to(self.device)  # (B,N,3)
        self.pcds_max = torch.tensor([[0.2, 0.2, 0.5]], dtype=torch.float32).to(self.device)
        self.pcds_min = torch.tensor([[-0.2, -0.2, -0.3]], dtype=torch.float32).to(self.device)
        self.pcds_center = torch.tensor([[0.00341, 0.0231, 0.7782]], dtype=torch.float32).to(self.device)


        for i in range(self.num_envs):
            # object_idx_this_env = i % len(self.object_code_list)
            object_idx_this_env =self.object_idxs[i]
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            max_agg_bodies = self.num_shadow_hand_bodies + self.num_object_bodies_list[object_idx_this_env] + 2
            max_agg_shapes = self.num_shadow_hand_shapes + self.num_object_shapes_list[object_idx_this_env] + 2

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # load shadow hand  for each env
            shadow_hand_actor = self._load_shadow_hand(env_ptr, i, shadow_hand_asset, shadow_hand_dof_props, shadow_hand_start_pose)

            # load object for each env

            scale = self.obj_trajs_info['obj_scale'][i]
            obj_rotmat = self.obj_trajs_info['obj_rotmat'][i]

            obj_half_height, pcd = self.get_obj_half_height(scale, obj_rotmat)
            # self.obj_half_height_list.append(obj_half_height)
            object_start_pose = self.set_obj_start_root_state(obj_half_height, obj_rotmat)
            object_handle = self._load_object(env_ptr, goal_start_pose, i, object_asset_dict, object_idx_this_env, object_start_pose, scale)
            # DexRep or pnG load object
            # assert len(self.object_code_list) == 1
            if self.use_dexrep:
                self.DexRepEncoder.load_cache_stl_file(
                    obj_idx=i,
                    obj_path=dexrep_load,
                    scale=scale)
            elif self.use_pnG:
                self.PnGEncoder.load_cache_stl_file(
                    obj_idx=i,
                    obj_path=dexrep_load,
                    scale=scale
                )
            elif self.use_geodex:
                self.GeoDexWrapper.load_cache_stl_file(
                    obj_idx=i,
                    obj_path=dexrep_load,
                    scale=scale
                )
            if self.use_dexrep:
                self.DexRepEncoder.load_batch_env_obj(object_idx_this_env)
            elif self.use_pnG:
                self.PnGEncoder.load_batch_env_obj(object_idx_this_env)
            elif self.use_geodex:
                self.GeoDexWrapper.load_batch_env_obj(object_idx_this_env)

            # add goal object
            # goal_asset_dict[id][scale_id]
            goal_handle = self.gym.create_actor(env_ptr, goal_asset_dict[object_idx_this_env], goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 1.0)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # ------------- set friction --------------
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            table_shape_props[0].friction = 1
            if self.cfg['env']['random_obj_mass_friction']:
                friction = random.uniform(0.1, 1.0)
            else:
                friction = self.cfg['env']['obj_friction']
            object_shape_props[0].friction = friction #1
            # object_shape_props[0].friction = self.cfg['env']['obj_friction'] #1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            # ---------set mass -------------
            if self.cfg["env"]["set_obj_mass"]:
                object_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                if len(object_body_props)>=1:
                    if self.cfg["env"]['random_obj_mass_friction']:
                        obj_mass = random.uniform(0.05, 0.5)
                    else:
                        obj_mass = self.cfg["env"]["obj_mass"]
                    for object_body_prop in object_body_props:
                        # object_body_prop.mass = self.cfg["env"]["obj_mass"]/len(object_body_props)
                        object_body_prop.mass = obj_mass/len(object_body_props)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, object_body_props)

            object_color = [90/255, 94/255, 173/255]
            self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
            table_color = [150/255, 150/255, 150/255]
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)
            self.objects.append(object_handle)


        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone()
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        self.goal_states[:, self.up_axis_idx] -= 0

        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)



    # def obj_pcd_transforms_from_state(self, obj_state, obj_verts):
    #     # B,D = obj_state.size()
    #
    #     obj_pos = obj_state[...,:3].reshape(-1,3)#(B,T,3)
    #     obj_quat = obj_state[...,3:][...,[3,0,1,2]].reshape(-1, 4)#(B*T,4)
    #     obj_rot = quaternion_to_matrix(obj_quat).float() #(B*T, 3, 3)
    #
    #
    #     obj_verts = obj_verts.unsqueeze(1).repeat(1,T,1,1).reshape(B*T, -1, 3) #(B*T,N,3)
    #     obj_verts = torch.bmm(obj_verts.float(), obj_rot.transpose(1, 2)) + obj_pos.unsqueeze(1)  # (B*T,N,3)
    #     obj_verts = obj_verts.reshape(B,T,-1, 3) # (B, T, N,3)
    #
    #     return obj_verts  # (B, T, N,3)

    def obj_pcd_transforms_from_state(self, obj_state, obj_verts):
        # B,D = obj_state.size()
        obj_pos = obj_state[...,:3].reshape(-1,3)#(B,3)
        obj_quat = obj_state[...,3:][...,[3,0,1,2]].reshape(-1,4)#(B,4)
        obj_rot = quaternion_to_matrix(obj_quat).float() #(B, 3, 3)
        obj_verts = torch.bmm(obj_verts.float(), obj_rot.transpose(1, 2)) + obj_pos.unsqueeze(1)  # (B,N,3)
        return obj_verts

    # def get_seq_object_mesh(self, obj_state, select_idxs=None):
    #     obj_scale = torch.tensor(self.obj_trajs_info['obj_scale']).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
    #     obj_rotmat = torch.tensor(self.obj_trajs_info['obj_rotmat'])  # (B,3,3)
    #
    #     if select_idxs is not None:
    #         obj_state = obj_state[select_idxs]
    #         obj_scale = obj_scale[select_idxs] # (B,1,1)
    #         obj_rotmat = obj_rotmat[select_idxs]  # (B,3,3)
    #
    #     B,T,D = obj_state.size()
    #
    #     obj_verts = torch.tensor(self.obj_mesh.vertices,dtype=torch.float32).unsqueeze(0).repeat(B,1,1) #(B, N, 3)
    #     obj_verts = torch.bmm(obj_verts, obj_rotmat.transpose(1,2))*obj_scale  #(B, N, 3)
    #
    #
    #     obj_verts = self.obj_pcd_transforms_from_state(obj_state, obj_verts)  # (B, T, N,3)
    #
    #     # obj_pos = obj_state[...,:3].reshape(-1,3)#(B,T,3)
    #     # obj_quat = obj_state[...,3:][...,[3,0,1,2]].reshape(-1, 4)#(B*T,4)
    #     # obj_rot = quaternion_to_matrix(obj_quat).float() #(B*T, 3, 3)
    #     #
    #     # obj_verts = obj_verts.unsqueeze(1).repeat(1,T,1,1).reshape(B*T, -1, 3) #(B*T,N,3)
    #     # obj_verts = torch.bmm(obj_verts.float(), obj_rot.transpose(1, 2)) + obj_pos.unsqueeze(1)  # (B*T,N,3)
    #     # obj_verts = obj_verts.reshape(B,T,-1, 3) # (B, T, N,3)
    #
    #     obj_seq_mesh_list = []
    #     for i in range(B):
    #         list_i = [trimesh.Trimesh(vertices=obj_verts[i,j].numpy(), faces=self.obj_mesh.faces) for j in range(T)]
    #         obj_seq_mesh_list.append(list_i)
    #     return obj_seq_mesh_list

    def get_seq_object_mesh(self, obj_state, select_idxs=None):
        obj_scale = torch.tensor(self.obj_trajs_info['obj_scale']).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        obj_rotmat = torch.tensor(self.obj_trajs_info['obj_rotmat'])  # (B,3,3)

        if select_idxs is not None:
            obj_state = obj_state[select_idxs]
            obj_scale = obj_scale[select_idxs] # (B,1,1)
            obj_rotmat = obj_rotmat[select_idxs]  # (B,3,3)

        B,T,D = obj_state.size()

        obj_verts = torch.tensor(self.obj_mesh.vertices,dtype=torch.float32).unsqueeze(0).repeat(B,1,1) #(B, N, 3)
        obj_verts = torch.bmm(obj_verts, obj_rotmat.transpose(1,2))*obj_scale  #(B, N, 3)


        obj_verts = self.obj_pcd_seq_transforms_from_state(obj_state, obj_verts)  # (B, T, N,3)


        obj_seq_mesh_list = []
        for i in range(B):
            list_i = [trimesh.Trimesh(vertices=obj_verts[i,j].numpy(), faces=self.obj_mesh.faces) for j in range(T)]
            obj_seq_mesh_list.append(list_i)
        return obj_seq_mesh_list

    def obj_pcd_seq_transforms_from_state(self, obj_state, obj_verts):
        B,T,D = obj_state.size()

        obj_pos = obj_state[...,:3].reshape(-1,3)#(B,T,3)
        obj_quat = obj_state[...,3:][...,[3,0,1,2]].reshape(-1, 4)#(B*T,4)
        obj_rot = quaternion_to_matrix(obj_quat).float() #(B*T, 3, 3)


        obj_verts = obj_verts.unsqueeze(1).repeat(1,T,1,1).reshape(B*T, -1, 3) #(B*T,N,3)
        obj_verts = torch.bmm(obj_verts.float(), obj_rot.transpose(1, 2)) + obj_pos.unsqueeze(1)  # (B*T,N,3)
        obj_verts = obj_verts.reshape(B,T,-1, 3) # (B, T, N,3)

        return obj_verts  # (B, T, N,3)

    def get_seq_obj_pcd(self, obj_state=None, select_idxs=None):
        obj_original_sample_points= self.get_object_sample_points(self.obj_mesh, point_num=512) #(N,512)

        obj_scale = torch.tensor(self.obj_trajs_info['obj_scale']).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        obj_rotmat = torch.tensor(self.obj_trajs_info['obj_rotmat'])  # (B,3,3)

        if select_idxs is not None:
            obj_state = obj_state[select_idxs]
            obj_scale = obj_scale[select_idxs] # (B,1,1)
            obj_rotmat = obj_rotmat[select_idxs]  # (B,3,3)

        B,T,D = obj_state.size()

        obj_sample_points = torch.tensor(obj_original_sample_points,dtype=torch.float32).unsqueeze(0).repeat(B,1,1) #(B, N, 3)
        obj_sample_points = torch.bmm(obj_sample_points, obj_rotmat.transpose(1,2))*obj_scale  #(B, N, 3)

        obj_sample_points = self.obj_pcd_transforms_from_state(obj_state, obj_sample_points) #(B,T,  N, 3)
        return obj_sample_points

    # def get_batch_obj_pcds(self,down_sample=True):
    #     if down_sample:
    #         sample_points = self.get_down_sample_points(self.obj_sample_points) #（512，3）
    #     else:
    #         sample_points = self.obj_sample_points
    #     obj_scale = torch.tensor(self.obj_trajs_info['obj_scale']).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
    #     obj_rotmat = torch.tensor(self.obj_trajs_info['obj_rotmat'])  # (B,3,3)
    #     B = obj_rotmat.size()[0]
    #
    #     obj_sample_points = torch.tensor(sample_points,dtype=torch.float32).unsqueeze(0).repeat(B,1,1) #(B, N, 3)
    #     obj_sample_points = torch.bmm(obj_sample_points, obj_rotmat.transpose(1,2))*obj_scale  #(B, N, 3)
    #     return obj_sample_points

    def get_batch_obj_pcds(self, down_sample=True):
        if down_sample:
            down_sampled_points_list = [self.get_down_sample_points(pcd) for pcd in
                                        self.obj_sample_points_list]  # list of (N, 3)
        else:
            down_sampled_points_list = self.obj_sample_points_list  # list of (N, 3)

        obj_scale = torch.tensor(self.obj_trajs_info['obj_scale']).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        obj_rotmat = torch.tensor(self.obj_trajs_info['obj_rotmat'])  # (B,3,3)
        object_idxs = torch.tensor(self.object_idxs)  # (B,)
        B = obj_rotmat.size(0)

        device = obj_rotmat.device

        obj_pcd_list = [
            torch.tensor(down_sampled_points_list[idx], dtype=torch.float32, device=device)
            for idx in object_idxs
        ]
        obj_sample_points = torch.stack(obj_pcd_list, dim=0)  # (B, N, 3)

        obj_sample_points = torch.bmm(obj_sample_points, obj_rotmat.transpose(1, 2)) * obj_scale  # (B, N, 3)

        return obj_sample_points

    def get_down_sample_points(self, pcd, n_num=512):
        indices = np.random.choice(pcd.shape[0], n_num, replace=False)
        return pcd[indices]

    def get_obs_pcds(self, obj_state, hand_state):
        obj_pcds =self.obj_pcd_transforms_from_state(obj_state,self.obj_init_obj_pcds) #(B,512,3)

        self.hand_model_forward(hand_state.cpu())
        hand_pcds,_ = self.hand_model.get_surface_points() #(B,512,3)

        obs_pcds = torch.cat([hand_pcds.to(obj_pcds.device),obj_pcds],dim=-2)#(B,1024,3)
        obs_pcds-=self.pcds_center
        obs_pcds = (obs_pcds-self.pcds_min)/(self.pcds_max-self.pcds_min)
        obs_pcds=obs_pcds*2-1
        return obs_pcds

    def get_obs_h2o(self,obj_state, hand_state):
        obj_pcds =self.obj_pcd_transforms_from_state(obj_state,self.obj_init_obj_pcds) #(B,2048,3)
        self.hand_model_forward(hand_state)
        hand_joints = self.hand_model.get_penetraion_keypoints() #(B,21,3)

        h2o_vec,h2o_dist =compute_h2o_minimum_vec(hand_joints, obj_pcds)

        h2o_vec = (h2o_vec-self.h2o_min)/(self.h2o_max-self.h2o_min)
        h2o_vec=h2o_vec*2-1
        return h2o_vec

    def get_seq_hand_pcd(self,hand_state, select_idxs=None, add_init_bias=True):
        if select_idxs is not None:
            hand_state = hand_state[select_idxs]
        B,T,D = hand_state.size()

        if add_init_bias:
            hand_state[...,2]+=self.init_hand_pos_z

        hand_state = hand_state.reshape(B*T, -1)#(B*T, 28)
        self.hand_model_forward(hand_state)
        pcd,_ = self.hand_model.get_surface_points() #(B,T,N,3)
        return pcd.reshape(B,T,-1,3)

    def detect_obj_pose_change(self, dist_thres=0.005):
        obj_state = self.get_object_state()
        distances = torch.norm(obj_state[:,:3]-self.init_obj_pos, dim=1)
        is_change_mask = (distances >= dist_thres).float()

        return is_change_mask


    def detect_h2o_close_enough(self,dist_thres=0.01):
        h2o_dist, obj_pcds = self.cal_h2o_distance()
        big_tips_dist = h2o_dist[:,-1] #(B,)

        #["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        big_tip_pos = self.fingertip_pos[:,-1,:] # (B,5,3)
        _, big_tips_dist2 = compute_h2o_minimum_vec(big_tip_pos, obj_pcds) #(B,)

        is_close_mask = (big_tips_dist <= dist_thres).float()
        return is_close_mask


    def cal_h2o_distance(self):
        hand_state = self.get_hand_state().unsqueeze(1)#（B，1，28）
        obj_state = self.get_object_state().unsqueeze(1) #（B，1，7）

        h2o_dist,obj_pcds = self.get_h2o_vector(hand_state, obj_state,return_dist=True) #(B,1, 21) (B,1,2048,3)
        return h2o_dist.squeeze(1), obj_pcds.squeeze(1)


    def apply_fingers_grip(self,actions, delta_angle=0.4,weight=1.5, tips_only=True):
        actions = actions.clone()

        if tips_only:
            actions[...,self.tips_idxs]+=delta_angle
            actions[...,self.big_tips_idx]-=delta_angle
        else:
            actions[...,6:]*=weight

        return actions


    # def get_pre_target_actions(self, iter):
    #     is_change_mask = self.detect_obj_pose_change()
    #     is_close_mask = self.detect_h2o_close_enough()
    #     check_mask = is_close_mask & is_change_mask
    #     self.check_mask+=check_mask
    #     check_mask = self.check_mask==1 #(判断第一次出现的帧，作为pre_target_actions)
    #
    #     self.iter_check_store[check_mask] = iter
    #     pre_target_actions = self.apply_fingers_grip(self.actions[check_mask])
    #
    #
    #     self.pre_target_actions[check_mask] = pre_target_actions
    #
    #
    #     actions = self.actions.clone()
    #
    #     achieve_target_mask = self.check_mask>=1
    #     actions[achieve_target_mask] = self.pre_target_actions[achieve_target_mask]
    #
    #     add_lift = 0.02*(iter - self.iter_check_store[achieve_target_mask]) #(N_check,)
    #     actions[achieve_target_mask,2]+=add_lift
    #
    #     return actions

    def get_pre_target_actions(self,actions, iter,previous_actions=None):
        is_change_mask = self.detect_obj_pose_change()
        is_close_mask = self.detect_h2o_close_enough()
        check_mask = is_close_mask.bool() | is_change_mask.bool()
        self.check_mask += check_mask
        check_mask = self.check_mask == 1

        self.iter_check_store[check_mask] = iter

        if previous_actions is not None:
            move_vectors = actions[:,:3]-previous_actions[:,:3] #(B,3)
            norms = move_vectors.norm(p=2,dim=-1,keepdim=True)
            move_vectors = move_vectors/(norms+1e-8)

            pre_target_actions = self.apply_fingers_grip(actions[check_mask],move_vectors[check_mask])
        else:
            pre_target_actions = self.apply_fingers_grip(actions[check_mask])

        self.pre_target_actions[check_mask] = pre_target_actions

        # actions = self.actions.clone()

        achieve_target_mask = self.check_mask >= 1
        actions[achieve_target_mask] = self.pre_target_actions[achieve_target_mask]

        add_lift = 0.02 * (iter - self.iter_check_store[achieve_target_mask])# (N_check,)


        actions[achieve_target_mask, 2] += add_lift

        return actions

    def hand_model_forward(self,hand_state):
        """
        hand_state (N, 28)
        """
        hand_pos = hand_state[...,:3].reshape(-1, 3)#(B*T, 3)
        hand_rot6d = euler_angles_to_matrix(hand_state[...,3:6],convention='XYZ').transpose(1,2).reshape(-1,9)[:,:6]#(B*T,6)
        hand_pose = torch.cat([hand_pos,hand_rot6d,hand_state[...,6:]],dim=1)
        self.hand_model.set_parameters(hand_pose)

    def get_seq_hand_mesh(self, hand_state,select_idxs=None, color='pink', return_id=0):
        """
        hand_state: (B,T,28)
        """

        if select_idxs is not None:
            hand_state = hand_state[select_idxs]
        B,T,D = hand_state.size()

        hand_state = hand_state.reshape(B*T, -1)#(B*T, 28)
        self.hand_model_forward(hand_state)


        hand_seq_mesh_list = []
        list_i = []
        for i in range(B*T):
            hand_mesh = self.hand_model.get_trimesh_data(i,color=color)
            hand_mesh = trimesh.util.concatenate(hand_mesh)
            if i%T==0:
                list_i=[]
                list_i.append(hand_mesh)
            else:
                list_i.append(hand_mesh)

            if i%T==T-1:
                hand_seq_mesh_list.append(list_i)



        return hand_seq_mesh_list


    def get_seq_hand_pcd(self,hand_state, select_idxs=None, add_init_bias=True, return_joints=True):
        if select_idxs is not None:
            hand_state = hand_state[select_idxs]
        B,T,D = hand_state.size()

        if add_init_bias:
            hand_state[...,2]+=self.init_hand_pos_z

        hand_state = hand_state.reshape(B*T, -1)#(B*T, 28)
        self.hand_model_forward(hand_state)
        pcd,_ = self.hand_model.get_surface_points() #(B*T,N,3)

        if return_joints:
            hand_joints = self.hand_model.get_penetraion_keypoints() #(B*T,21,3)
            return pcd.reshape(B,T,-1,3), hand_joints.reshape(B, T, -1, 3)
        return pcd.reshape(B,T,-1,3), None


    def get_hand_joints(self, hand_state, select_idxs=None, add_init_bias=True):
        if select_idxs is not None:
            hand_state = hand_state[select_idxs]
        B,T,D = hand_state.size()

        if add_init_bias:
            hand_state[...,2]+=self.init_hand_pos_z

        hand_state = hand_state.reshape(B*T, -1)#(B*T, 28)
        self.hand_model_forward(hand_state)
        hand_joints = self.hand_model.get_penetraion_keypoints()  # (B*T,21,3)
        return hand_joints.reshape(B,T,-1,3)

    def get_h2o_vector(self, hand_state, obj_state, select_idxs=None, add_init_bias=True, return_dist=False):

        device = obj_state.device
        obj_scale = torch.tensor(self.obj_trajs_info['obj_scale'], device=device).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        obj_rotmat = torch.tensor(self.obj_trajs_info['obj_rotmat'], device=device)  # (B,3,3)
        obj_idxs = torch.tensor(self.object_idxs, device=device)
        if select_idxs is not None:
            obj_state = obj_state[select_idxs]
            obj_scale = obj_scale[select_idxs] # (B,1,1)
            obj_rotmat = obj_rotmat[select_idxs]  # (B,3,3)
            obj_idxs = self.object_idxs[select_idxs]

        B,T,D = obj_state.size()
        if len(self.object_code_list) == 1:
            obj_pcds = torch.tensor(self.obj_sample_points,dtype=torch.float32).unsqueeze(0).repeat(B,1,1) #(B, 2048, 3)
        else:
            obj_pcd_list = [torch.tensor(self.obj_sample_points_list[idx], dtype=torch.float32, device=device) for idx in obj_idxs]
            obj_pcds = torch.stack(obj_pcd_list, dim=0)
        obj_pcds = torch.bmm(obj_pcds, obj_rotmat.transpose(1,2))*obj_scale  #(B, 2048, 3)
        obj_pcds = self.obj_pcd_transforms_from_state(obj_state, obj_pcds) #(B,T,  2048, 3)

        hand_joints = self.get_hand_joints(hand_state.cpu(), select_idxs, add_init_bias=add_init_bias).to(device) #(B,T,N,3)


        h2o_vec,h2o_dist =compute_h2o_minimum_vec(hand_joints.reshape(B*T,-1,3), obj_pcds.reshape(B*T,-1,3))
        if return_dist:
            return h2o_dist.reshape(B, T, -1), obj_pcds

        return h2o_vec.reshape(B,T,-1,3)



    def html_save(self, obj_seq_state, hand_seq_state,success_idx=None,key_str='', add_init_bias=True,extra_hand_state=None):
        """
        obj_seq_state(B,T,7)
        hand_seq_state(B,T,28)
        """
        _,T, _ = obj_seq_state.size()

        if add_init_bias:
            hand_seq_state[...,2]+=self.init_hand_pos_z

        obj_seq_meshes_list = self.get_seq_object_mesh(obj_seq_state, success_idx) #list (B,T)
        hand_seq_mesh_list = self.get_seq_hand_mesh(hand_seq_state, success_idx)  #list (B,T)
        B = len(obj_seq_meshes_list)

        extra_hand_seq_mesh_list=None
        if extra_hand_state is not None:
            extra_hand_state[...,2]+=self.init_hand_pos_z
            extra_hand_seq_mesh_list = self.get_seq_hand_mesh(extra_hand_state, success_idx,color='red')

        for i in range(B):
            idx = success_idx[i]
            obj_seq_mesh_i, hand_seq_mesh_i = obj_seq_meshes_list[i],  hand_seq_mesh_list[i]
            if isinstance(extra_hand_seq_mesh_list,list):
                extra_seq_meshes_list_i = extra_hand_seq_mesh_list[i]

            name = self.object_code_list[0]+'_seq{}_'.format(idx)+key_str
            if extra_hand_seq_mesh_list is not None:
                html_antmation_save(obj_seq_mesh_i, hand_seq_mesh_i, extra_seq_meshes_list_i, name=name)
            else:
                html_antmation_save(obj_seq_mesh_i, hand_seq_mesh_i, name=name)

            a=1


    def get_object_state(self):
        obj_pos = self.root_state_tensor.view(self.num_envs, -1, 13)[:, self.objects[0], :3]  # (B,3)
        obj_rot = self.root_state_tensor.view(self.num_envs, -1, 13)[:, self.objects[0], 3:7]  # (B,4)

        return torch.cat([obj_pos, obj_rot], dim=1) #(B, 7)

    def get_hand_state(self, add_init_bias=True):
        # hand_pose_params= []
        # for i in range(self.num_envs):
        #     hand_pose_params.append(self.gym.get_actor_dof_states(self.envs[i], self.shadow_hands[i], gymapi.STATE_POS)['pos'])# (28,)
        # hand_pose_params = np.stack(hand_pose_params,axis=0)
        hand_pose_params = self.shadow_hand_dof_pos.clone()
        if add_init_bias:
            hand_pose_params[:, 2] += self.init_hand_pos_z

        return hand_pose_params

    def get_object_sample_points(self, obj_mesh, point_num=2048):
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(obj_mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(obj_mesh.faces)
        pcd = mesh_o3d.sample_points_poisson_disk(point_num)
        pcd = np.asarray(pcd.points)
        return pcd

    def get_obj_half_height(self,object_scale, object_rotmat, object_id=0):
        obj_sample_points = self.obj_sample_points_list[object_id]

        pcd = np.matmul(obj_sample_points, object_rotmat.T)* object_scale
        min_z = np.min(pcd[:, 2])
        obj_half_height = min_z #* object_scale
        return obj_half_height, pcd
    def set_obj_start_root_state(self, obj_half_height,object_rotmat):
        r = R.from_matrix(object_rotmat)
        rot_quat = r.as_quat()

        object_z = self.table_height - obj_half_height+0.005
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, object_z)  # gymapi.Vec3(0.0, 0.0, 0.72)
        # object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        # object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        object_start_pose.r  = gymapi.Quat(rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])
        return object_start_pose

    def _load_object(self, env_ptr, goal_start_pose, i, object_asset_dict, object_idx_this_env, object_start_pose,
                     scale):
        object_handle = self.gym.create_actor(env_ptr, object_asset_dict[object_idx_this_env], object_start_pose,
                                              "object", i, 0, 0)
        self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                       object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                       object_start_pose.r.w,
                                       0, 0, 0, 0, 0, 0])
        self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                     goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                     goal_start_pose.r.w,
                                     0, 0, 0, 0, 0, 0])
        object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
        self.object_indices.append(object_idx)
        self.gym.set_actor_scale(env_ptr, object_handle, scale)
        return object_handle

    def _load_shadow_hand(self, env_ptr, i, shadow_hand_asset, shadow_hand_dof_props, shadow_hand_start_pose):
        shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
        self.hand_start_states.append(
            [shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
             shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z,
             shadow_hand_start_pose.r.w,
             0, 0, 0, 0, 0, 0])
        self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
        hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
        self.hand_indices.append(hand_idx)
        # randomize colors and textures for rigid body
        num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, shadow_hand_actor)
        hand_color = [147 / 255, 215 / 255, 160 / 255]
        hand_rigid_body_index = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19, 20],
                                 [21, 22, 23, 24, 25]]
        for n in self.agent_index[0]:
            for m in n:
                for o in hand_rigid_body_index[m]:
                    self.gym.set_rigid_body_color(env_ptr, shadow_hand_actor, o, gymapi.MESH_VISUAL,
                                                  gymapi.Vec3(*hand_color))
        # create fingertip force-torque sensors
        # if self.obs_type == "full_state" or self.asymmetric_obs:
        self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)
        return shadow_hand_actor

    def _load_table_asset(self):
        table_dims = gymapi.Vec3(1, 1, self.table_height)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())
        return table_asset, table_dims


    def get_object_name(self, object_code):
        object_code = object_code.split('.')[0]
        if 'ddg-gd' in object_code or 'ddg-kit' in object_code:
            object_name = object_code.split('_')[1]
        elif 'ddg-ycb' in object_code or "mujoco" in object_code:
            object_name = object_code.split('_')[-1]
        else:
            object_name = object_code.split('-')[1]

        object_name = object_name.lower()

        return object_name

    def _load_object_asset(self, assets_path):
        object_asset_dict = {}
        goal_asset_dict = {}
        self.num_object_bodies_list = []
        self.num_object_shapes_list = []
        # mesh_path = osp.join(assets_path, 'meshdatav3_scaled')
        self.asset_root = self.cfg["env"]["asset"]["assetRoot"]
        self.obj_asset_root = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj"]
        self.raw_obj_asset_root = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj_raw"]
        for object_id, object_code in enumerate(self.object_code_list):
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            if self.cfg["env"]["set_obj_mass"]==False:
                object_asset_options.density = 1
            object_asset_options.fix_base_link = False
            # object_asset_options.disable_gravity = True
            object_asset_options.use_mesh_materials = True
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            object_asset = None
            object_asset_file = "coacd_1.urdf"
            if self.cfg['env']['obj_type'] == 'seen' or 'one':
                # object_asset = self.gym.load_asset(self.sim, self.obj_asset_root + f'{object_code}' + "/coacd", object_asset_file, object_asset_options)
                object_asset = self.gym.load_asset(self.sim, '../assets/meshdata/' + f'{object_code}' + "/coacd",
                                                   object_asset_file, object_asset_options)
            if object_asset is None:
                print(object_code)
            assert object_asset is not None

            object_asset_options.disable_gravity = True
            goal_asset = self.gym.create_sphere(self.sim, 0.005, object_asset_options)

            # self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
            # self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
            self.num_object_bodies_list.append(self.gym.get_asset_rigid_body_count(object_asset))
            self.num_object_shapes_list.append(self.gym.get_asset_rigid_shape_count(object_asset))
            # set object dof properties
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset)
            self.object_dof_lower_limits = []
            self.object_dof_upper_limits = []

            for i in range(self.num_object_dofs):
                self.object_dof_lower_limits.append(object_dof_props['lower'][i])
                self.object_dof_upper_limits.append(object_dof_props['upper'][i])

            self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
            self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)
            object_asset_dict[object_id] = object_asset
            goal_asset_dict[object_id] = goal_asset
        return goal_asset_dict, object_asset_dict

    def _load_shadow_hand_asset(self):
        asset_root = "../../assets"
        shadow_hand_asset_file = "mjcf_free/open_ai_assets/hand/shadow_hand.xml"
        table_texture_files = "../assets/textures/texture_wood_brown_1033760.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)
        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)
        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 1
        asset_options.linear_damping = 50
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)
        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)
        # tendon set up

        limit_stiffness = 10
        t_damping = 5

        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)
        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in
                              range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in
                                     actuated_dof_names]
        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        shadow_hand_dof_props['damping'][6:]*=self.cfg['env']['damping_w']
        shadow_hand_dof_props['stiffness'][6:]*=self.cfg['env']['stiffness_w']

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)
        return shadow_hand_asset, shadow_hand_dof_props, table_texture_handle

    def clean_sim(self):
        self.camera_rgb_tensor_list=[]

        self.num_object_bodies_list=[]
        self.num_object_shapes_list = []
        self.object_code_list = []
        if self.headless == False:
            self.gym.destroy_viewer(self.my_viewer)


        self.gym.destroy_sim(self.sim)


    def compute_reward(self, actions, id=-1):
        self.dof_pos = self.shadow_hand_dof_pos
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.object_init_z,
            self.id, self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
            self.goal_pos, self.goal_rot,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_lf_pos, self.right_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,self.goal_cond
        )

        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))


    def sim_refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def compute_observations(self):
        # TODO:using dexrep
        self.sim_refresh()

        # if self.obs_type == "full_state" or self.asymmetric_obs:
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_handle_pos = self.object_pos  ##+ quat_apply(self.object_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.object_back_pos = self.object_pos + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        idx = self.hand_body_idx_dict['palm']
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        if self.use_dexrep or self.use_pnG or self.use_obs_pcds:
            self.dexrep_hand_state = self.rigid_body_states[:, self.dexrep_hand_indices, :].view(self.num_envs, -1, 13)
            self.dexrep_hand_pos = self.dexrep_hand_state[:, :, 0:3]
            self.dexrep_hand_vel = self.dexrep_hand_state[:, :, 7:13]
            # compute fingertip
            idx = 0
            self.right_hand_ff_pos, self.right_hand_ff_rot = self.dexrep_hand_state[:, idx, 0:3], self.dexrep_hand_state[:, idx, 3:7]
            self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)

            idx = 1
            self.right_hand_mf_pos, self.right_hand_mf_rot = self.dexrep_hand_state[:, idx, 0:3], self.dexrep_hand_state[:, idx, 3:7]
            self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)

            idx = 2
            self.right_hand_rf_pos = self.dexrep_hand_state[:, idx, 0:3]
            self.right_hand_rf_rot = self.dexrep_hand_state[:, idx, 3:7]
            self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)

            idx = 3
            self.right_hand_lf_pos = self.dexrep_hand_state[:, idx, 0:3]
            self.right_hand_lf_rot = self.dexrep_hand_state[:, idx, 3:7]
            self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)

            idx = 4
            self.right_hand_th_pos = self.dexrep_hand_state[:, idx, 0:3]
            self.right_hand_th_rot = self.dexrep_hand_state[:, idx, 3:7]
            self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)
            # concatenate
            fingertip_pos = torch.cat(
                (self.right_hand_ff_pos.unsqueeze(-2),
                 self.right_hand_mf_pos.unsqueeze(-2),
                 self.right_hand_rf_pos.unsqueeze(-2),
                 self.right_hand_lf_pos.unsqueeze(-2),
                 self.right_hand_th_pos.unsqueeze(-2)),
                dim=1
            )
            self.dexrep_hand_pos = torch.cat(    # expected [B, 20, 3]
                (fingertip_pos, self.dexrep_hand_pos),
                dim=1
            )

        # self.fingertip_state = self.rigid_body_states[self.fingertip_indices].view(self.num_envs, -1, 13)
        # self.fingertip_pos = self.fingertip_state[:, :, 0:3]
        # self.fingertip_ori = self.fingertip_state[:, :, 3:7]
        # self.fingertip_lin_vel = self.fingertip_state[:, :, 7:10]
        # self.fingertip_ang_vel = self.fingertip_state[:, :, 10:13]
        # self.fingertip_vel = self.fingertip_state[:, :, 7:13]
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        base_state = self.compute_full_state()
        base_state = torch.clamp(base_state, -self.cfg["env"]["clip_observations"],
                                 self.cfg["env"]["clip_observations"])

        if self.obs_type in ['DexRep']:
            assert self.use_dexrep
            dexrep_obs = self.DexRepEncoder.pre_observation(
                obj_pos=self.object_pos,
                obj_rot=self.object_rot,
                hand_pos=self.dexrep_hand_state[:, 11, 0:3].squeeze(dim=1),
                hand_rot=self.dexrep_hand_state[:, 11, 3:7].squeeze(dim=1),
                joints_sate=self.dexrep_hand_pos,
                clip_range=self.cfg["env"]["clip_observations"]
            )
            # dexrep_obs = torch.clamp(dexrep_obs, -self.cfg["env"]["clip_observations"],
            #                      self.cfg["env"]["clip_observations"])
            self.obs_buf = torch.cat(
                (base_state, dexrep_obs),
                dim=1
            )

        elif self.obs_type in ['obs_pcds']:
            hand_state = self.get_hand_state()
            obj_state = self.get_object_state()
            self.obs_buf = self.get_obs_pcds(obj_state, hand_state)  # (B,512,3)

            self.unactions = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits,
                                     self.shadow_hand_dof_upper_limits)  # (B,28)

        elif self.obs_type in ['obs_h2o']:
            hand_state = self.get_hand_state()
            obj_state = self.get_object_state()
            self.obs_buf = self.get_obs_h2o(obj_state, hand_state)  # (B,21,3)
            self.unactions = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits,
                                     self.shadow_hand_dof_upper_limits)  # (B,28)

        else:
            raise AttributeError(f'{self.obs_type} not include..')

    def get_unpose_quat(self):
        if self.repose_z:
            self.unpose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                -self.z_theta,
            )
        return

    def unpose_point(self, point):
        if self.repose_z:
            return self.unpose_vec(point)
            # return self.origin + self.unpose_vec(point - self.origin)
        return point

    def unpose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.unpose_z_theta_quat, vec)
        return vec

    def unpose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.unpose_z_theta_quat, quat)
        return quat

    def unpose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.unpose_point(state[:, 0:3])
            state[:, 3:7] = self.unpose_quat(state[:, 3:7])
            state[:, 7:10] = self.unpose_vec(state[:, 7:10])
            state[:, 10:13] = self.unpose_vec(state[:, 10:13])
        return state

    def get_pose_quat(self):
        if self.repose_z:
            self.pose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                self.z_theta,
            )
        return

    def pose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.pose_z_theta_quat, vec)
        return vec

    def pose_point(self, point):
        if self.repose_z:
            return self.pose_vec(point)
            # return self.origin + self.pose_vec(point - self.origin)
        return point

    def pose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.pose_z_theta_quat, quat)
        return quat

    def pose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.pose_point(state[:, 0:3])
            state[:, 3:7] = self.pose_quat(state[:, 3:7])
            state[:, 7:10] = self.pose_vec(state[:, 7:10])
            state[:, 10:13] = self.pose_vec(state[:, 10:13])
        return state

    def compute_full_state(self, asymm_obs=False):

        self.get_unpose_quat()
        obs_buf = torch.zeros((self.num_envs, 222), device=self.device, dtype=torch.float)
        # unscale to (-1，1)
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##

        # 0：84
        obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                               self.shadow_hand_dof_lower_limits,
                                                               self.shadow_hand_dof_upper_limits)
        obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        obs_buf[:,2 * self.num_shadow_hand_dofs:3 * self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :]
        fingertip_obs_start = 3 * self.num_shadow_hand_dofs
        aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        for i in range(5):
            aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])
        # 84:149: ft states
        obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = aux

        # 149:179: ft sensors: do not need repose
        obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]

        hand_pose_start = fingertip_obs_start + 95
        # 179:185: hand_pose
        obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
        euler_xyz = get_euler_xyz(self.unpose_quat(self.hand_orientations[self.hand_indices, :]))
        obs_buf[:, hand_pose_start + 3:hand_pose_start + 4] = euler_xyz[0].unsqueeze(-1)
        obs_buf[:, hand_pose_start + 4:hand_pose_start + 5] = euler_xyz[1].unsqueeze(-1)
        obs_buf[:, hand_pose_start + 5:hand_pose_start + 6] = euler_xyz[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        # 185:209: action
        aux = self.actions[:, :24] #！！！ 分情况 bc_infer和 obs_extact

        # aux[:, 0:3] = self.unpose_vec(aux[:, 0:3])
        # aux[:, 3:6] = self.unpose_vec(aux[:, 3:6])
        # obs_buf[:, action_obs_start:action_obs_start + 24] = aux
        if self.cfg['env']['env_mode']=='extract_obs' or self.id==-1:
            obs_buf[:, action_obs_start:action_obs_start + 24] = unscale(aux,
                                                                         self.shadow_hand_dof_lower_limits[:24],
                                                                         self.shadow_hand_dof_upper_limits[:24])
        else:
            obs_buf[:, action_obs_start:action_obs_start + 24] = aux

        obj_obs_start = action_obs_start + 24  # 144
        # 209:222 object_pose, goal_pos
        obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.unpose_point(self.object_pose[:, 0:3])
        obs_buf[:, obj_obs_start + 3:obj_obs_start + 7] = self.unpose_quat(self.object_pose[:, 3:7])
        obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.unpose_vec(self.object_linvel)
        obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.unpose_vec(self.object_angvel)
        # obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.unpose_vec(self.goal_pos - self.object_pos)

         # 207:236 goal
        # hand_goal_start = obj_obs_start + 16
        # obs_buf[:, hand_goal_start:hand_goal_start + 3] = self.delta_target_hand_pos
        # obs_buf[:, hand_goal_start + 3:hand_goal_start + 7] = self.delta_target_hand_rot
        # obs_buf[:, hand_goal_start + 7:hand_goal_start + 29] = self.delta_qpos

        # 236: visual feature
        # visual_feat_start = hand_goal_start + 29

        # 236: 300: visual feature
        # obs_buf[:, visual_feat_start:visual_feat_start + 64] = 0.1 * self.visual_feat_buf

        return obs_buf

    def reset_target_pose(self, env_ids, apply_reset=False):

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]

        # self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]  # + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]

        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):

        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_shadow_hand_dofs]

        # pos = self.shadow_hand_default_dof_pos  # + self.reset_dof_pos_noise * rand_delta

        #设置第一帧为起始姿态
        first_action = self.grasp_seqs[:, 0, :]
        first_action[:,6:]=0.
        self.shadow_hand_dof_pos[env_ids, :] = first_action[env_ids,:]

        pos = first_action[env_ids,:]


        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel #+ self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]

        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        all_indices = torch.unique(torch.cat([all_hand_indices, self.object_indices[env_ids], self.table_indices[env_ids], ]).to(torch.int32))  ##

        self.hand_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 0:3]
        self.hand_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 3:7]
        self.hand_linvels[hand_indices.to(torch.long), :] = 0
        self.hand_angvels[hand_indices.to(torch.long), :] = 0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        # self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot  # reset object rotation
        # self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              self.object_indices[env_ids],
                                              self.goal_object_indices[env_ids],
                                              self.table_indices[env_ids], ]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))

        if self.random_time:
            self.random_time = False
            self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        else:
            self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        return first_action

    def pre_physics_step(self, actions, id):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envspre_physics_step
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            first_actions = self.reset(env_ids, goal_env_ids)
            actions = first_actions

        self.get_pose_quat()
        # actions[:, 0:3] = self.pose_vec(actions[:, 0:3])
        # actions[:, 3:6] = self.pose_vec(actions[:, 3:6])
        self.actions = actions.clone().to(self.device)

        if self.cfg['env']['env_mode'] == 'extract_obs' or id==-1:
            self.cur_targets[:] = self.actions
            self.cur_targets[:] = self.act_moving_average * self.cur_targets[:] + (1.0 - self.act_moving_average) * self.prev_targets[:]
            self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        # if self.use_relative_control:
        #     targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
        #     self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        # else:
        elif self.cfg['env']['env_mode'] == 'bc_env_infer':
            # self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, :],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:] = scale(self.actions[:, :],self.shadow_hand_dof_lower_limits,self.shadow_hand_dof_upper_limits)
            # -------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-------------------
            # self.cur_targets[:] = scale(self.actions[:, :],self.shadow_hand_dof_lower_limits,self.shadow_hand_dof_upper_limits)/self.unscale_actions_rescale

            self.cur_targets[:] = self.act_moving_average * self.cur_targets[:] + (1.0 - self.act_moving_average) * self.prev_targets[:]
            self.cur_targets[:] = tensor_clamp(self.cur_targets[:],self.shadow_hand_dof_lower_limits,self.shadow_hand_dof_upper_limits)


            # self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
            # self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000

            # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
            #                                         gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

            self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                    gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions, self.id)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.object_back_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.goal_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_pos[i], self.right_hand_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_th_pos[i], self.right_hand_th_rot[i])

                # self.add_debug_lines(self.envs[i], self.left_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_th_pos[i], self.right_hand_th_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])


#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
def compute_hand_reward(
        object_init_z,
        id: int, object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, current_successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
        right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    reward = 0


    resets = reset_buf

    # Find out which envs hit the goal and update successes count
    # resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = resets
    # successes = torch.where(goal_dist <= 0.10, torch.ones_like(successes), successes)
    # successes = torch.where(object_pos[:, 2] >= target_pos[:, 2], torch.ones_like(successes), successes)


    # successes_ext =torch.where(object_pos[:,2]<=1.7, torch.ones_like(successes), torch.zeros_like(successes))
    successes_ext = torch.where(
        (object_pos[:, 0] >= -1.5) & (object_pos[:, 0] <= 1.5) &
        (object_pos[:, 1] >= -1.5) & (object_pos[:, 1] <= 1.5) &
        (object_pos[:, 2] < 2.0),
        torch.ones_like(successes),
        torch.zeros_like(successes)
    )
    successes =torch.where(goal_dist <= 0.12, torch.ones_like(successes),successes)
    successes = torch.where(object_pos[:, 2] >= target_pos[:, 2], torch.ones_like(successes), successes)
    successes = successes_ext * successes

    # successes = torch.where(object_pos[:, 2] <= target_pos[:, 2]+1, torch.ones_like(successes), successes)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    current_successes = torch.where(resets==1, successes, current_successes)
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes


def compute_hand_reward_rl(
        object_init_z,
        id: int, object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, current_successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
        right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.norm(object_handle_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.where(right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist)

    right_hand_finger_dist = (torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos - right_hand_mf_pos, p=2, dim=-1)+ torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(
                object_handle_pos - right_hand_lf_pos, p=2, dim=-1) + torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1))
    right_hand_finger_dist = torch.where(right_hand_finger_dist >= 3.0, 3.0 + 0 * right_hand_finger_dist,right_hand_finger_dist)
    lowest = object_pos[:, 2]


    flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
    goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
    goal_hand_rew = torch.where(flag == 2, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)

    hand_up = torch.zeros_like(right_hand_finger_dist)
    hand_up = torch.where(lowest >= 0.630, torch.where(flag == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
    hand_up = torch.where(lowest >= 0.80, torch.where(flag == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

    flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)

    reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus


    resets = reset_buf

    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = resets
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    current_successes = torch.where(resets.bool(), successes, current_successes)
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes




@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
