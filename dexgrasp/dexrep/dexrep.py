import os.path
import open3d as o3d
import numpy as np
import copy
import torch
from dexrep.utils import penetration, pc_normalize_tensor, pc_normalize2_tensor, denormalize, DotDict
from dexrep.pointnet_model.model_rec import ShapeNetAutoEncoder
from pytorch3d.transforms import quaternion_to_matrix
import dexrep.utils as util


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
# device = torch.device('cpu')

class cv_space(object):
    def __init__(self, args, sensor_type="all_pointnetL_pre", device="cuda:0"):
        # ts=time.time()
        # self.length = 0.1
        # # length_per_cube = 0.01
        # self.length_per_cube = 0.01
        self.device = device
        self.args = args

        # length_per_cube = 0.01
        self.length_per_cube = self.args["dexrep"]["ambient_voxel_length"]
        self.length = self.length_per_cube * 10
        self.all_cubes_center = self.create_grid(self.length, self.length_per_cube)
        self.dis_max_range = self.args["dexrep"]["dis_max_range"]
        print("ambient_voxel_length: ", self.length_per_cube)
        print("dis_max_range: ", self.dis_max_range)
        # print('grid:', time.time()-ts)
        # self.VG.show()

        # self.reservoir = extract_T_info() if save_data else None
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))
        # mesh_dir = os.path.dirname(self.curr_dir)
        self.sensot_type = sensor_type
        # if 'pre' in self.sensot_type:
        if self.sensot_type not in ["surf", "surf2g", "surfv2g", "surfvv2g", "surfh2g"]:
            self.load_pcd_extractor()
        self.obj_info = dict()

        self.batch_split_num = 50

        self.all_cubes_center = None


    def create_grid(self, length, length_per_cube, dtype=torch.float32):
        start = -(length - length_per_cube) / 2
        end = (length - length_per_cube) / 2
        x = torch.linspace(start, end, int(length / length_per_cube), dtype=dtype)
        y = torch.linspace(start, end, int(length / length_per_cube), dtype=dtype)
        z = torch.linspace(start, end, int(length / length_per_cube), dtype=dtype)
        xx, yy, zz = torch.meshgrid(x, y, z)
        positions = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

        return positions

    def load_pcd_extractor(self):
        checkpoint_path = 'pointnet_model/epo_180_REC_SPnetDenseEncoder_shapenet55_normrot512.pt'
        checkpoint_path = os.path.join(self.curr_dir, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.PN = ShapeNetAutoEncoder()
        self.PN.load_state_dict(checkpoint)
        self.PN = self.PN.to(self.device)
        self.PN.requires_grad_(False)
        self.PN.eval()

    def import_mesh(self, stl_dir):
        # self.obj_info['verts_sample_id'] = None
        self.obj_info = self.obj_info
        if self.obj_info['verts_sample_id'] is not None:
            mesh_obj = o3d.geometry.TriangleMesh()
            mesh_obj.vertices = o3d.utility.Vector3dVector(self.obj_info['verts'])
            mesh_obj.triangles = o3d.utility.Vector3iVector(self.obj_info['faces'])
            mesh_obj.compute_vertex_normals(normalized=True)
            mesh_obj.compute_triangle_normals(normalized=True)
            self.obj_info['verts_sample'] = torch.from_numpy(np.asarray(mesh_obj.vertices)[self.obj_info['verts_sample_id']]).float().to(self.device)
            self.obj_info['verts_normal'] = torch.from_numpy(np.asarray(mesh_obj.vertex_normals)[self.obj_info['verts_sample_id']]).float().to(self.device)
            self.obj_info['verts_sample_id'] = torch.LongTensor(self.obj_info['verts_sample_id']).to(self.device)
        else:
            self.obj_info['verts_sample'] = torch.from_numpy(self.obj_info['verts_sample']).float().to(self.device)
            self.obj_info['verts_normal'] = torch.from_numpy(self.obj_info['verts_normal']).float().to(self.device)

        self.obj_info_ori = copy.deepcopy(self.obj_info)


    def get_perception_data(self, pos, quat, joint_sites, obj_verts, obj_norms):
        # pos = torch.from_numpy(pos).float().to(self.device)
        # quat = torch.from_numpy(quat).float().to(self.device)
        # rot = quaternion_to_matrix(quat)
        #
        # joint_sites = torch.from_numpy(joint_sites).float().to(self.device)
        #
        # obj_verts = torch.from_numpy(obj_verts).float().to(self.device)
        # obj_norms = torch.from_numpy(obj_norms).float().to(self.device)
        pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
        quat = torch.tensor(quat, dtype=torch.float32).to(self.device)
        rot = quaternion_to_matrix(quat)  # (N, 30, 3, 3)

        joint_sites = torch.tensor(joint_sites, dtype=torch.float32).to(self.device)
        obj_verts = torch.tensor(obj_verts, dtype=torch.float32).to(self.device)
        obj_norms = torch.tensor(obj_norms, dtype=torch.float32).to(self.device)

        obj_rot_mat = rot[-2]
        obj_trans = pos[-2]
        self.obj_info['verts_sample'] = torch.matmul(obj_verts, obj_rot_mat.T) + obj_trans.reshape(1, 3)
        self.obj_info['verts_normal'] = torch.matmul(obj_norms, obj_rot_mat.T)
        obj_mesh=DotDict(self.obj_info)
        # self.assemble_and_move(pos, quat)
        if self.sensot_type == 'all_pointnetL_pre':
            # ts = time.time()
            signed_dis= self.singed_distance_sensors(obj_mesh, joint_sites)
            # print('sd', time.time()-ts)
            # ts = time.time()
            ambient_shape = self.ambient_sensors(rot[11], pos[11], obj_mesh)  # the root joints of middle finger
            # print('am', time.time() - ts)
            sensors = torch.cat([signed_dis.flatten(), ambient_shape])
            # ts = time.time()
            obj_sampled_verts = obj_mesh.verts_sample
            dis = torch.cdist(joint_sites, obj_sampled_verts)
            dis_min_idx = torch.argmin(dis, dim=-1)
            # self.instance(obj_sampled_verts)
            pn_feat = self.get_demon_pointnet_feature(obj_sampled_verts, dis_min_idx)
            # print('pn', time.time() - ts)
            return sensors.cpu().numpy(), pn_feat[1024:].cpu().numpy()

        else:
            raise AssertionError('sensor type is not included yet')

    def get_batch_perception_data(self, pos, quat, joint_sites, hand_pos, hand_quat, obj_verts, obj_norms,
                                  goal_pos=None, goal_quat=None):
                                  # mesh_names=None, sdfs=None):
        """ N x pos/quat/..."""
        # pos = torch.tensor(pos, dtype=torch.float32).to(self.device)    # (N, 30, 3)
        # quat = torch.tensor(quat, dtype=torch.float32).to(self.device)  # (N, 30, 4)
        rot = quaternion_to_matrix(torch.roll(quat, 1, dims=1))                  # (N, 30, 3, 3)
        hand_rot = quaternion_to_matrix(torch.roll(hand_quat, 1, dims=1))

        # joint_sites = torch.tensor(joint_sites, dtype=torch.float32).to(self.device)    # (N, 20, 3)
        # obj_verts = torch.tensor(obj_verts, dtype=torch.float32).to(self.device)   # (N, 2048, 3)
        # obj_norms = torch.tensor(obj_norms, dtype=torch.float32).to(self.device)   # (N, 2048, 3)

        # Env_Num = pos.shape[0]
        # obj_rot_mat = rot[:, -2, :, :].squeeze(dim=1)    # (N, 1, 3, 3) -> (N, 3, 3)
        # obj_trans = pos[:, -2, :].unsqueeze(dim=-2)      # (N, 1, 3)
        obj_rot_mat = rot  # (N, 3, 3)
        obj_trans = pos.unsqueeze(dim=-2) # (N, 1, 3)

        if self.sensot_type in ['dexrep_double', "dexrep_toGoal", "dexrep_VtoGoal", "dexrep_VVtoGoal", "dexrep_HtoGoal",
                                "surf2g", "surfv2g", "surfvv2g", "surfh2g"]:
            goal_rot = quaternion_to_matrix(torch.roll(goal_quat, 1, dims=1))
            goal_rot_mat = goal_rot
            goal_trans = goal_pos.unsqueeze(dim=-2)

        if self.sensot_type in ['all_pointnetL_pre', 'pointnetG_pre', 'dexrep_double', "surf"]:
            self.obj_info['verts_sample'] = torch.matmul(torch.clone(obj_verts), obj_rot_mat.transpose(-1, -2)) + obj_trans
            self.obj_info['verts_normal'] = torch.matmul(torch.clone(obj_norms), obj_rot_mat.transpose(-1, -2))
            self.obj_info["rot_mat"] = obj_rot_mat
            self.obj_info["trans_mat"] = obj_trans
        elif self.sensot_type in ["dexrep_toGoal", "dexrep_VtoGoal", "dexrep_VVtoGoal", "dexrep_HtoGoal", "surf2g", "surfv2g", "surfvv2g", "surfh2g"]:
            self.obj_info['verts_sample'] = obj_verts
            self.obj_info['verts_normal'] = obj_norms
            self.obj_info["rot_mat"] = obj_rot_mat
            self.obj_info["trans_mat"] = obj_trans
            self.obj_info["goal_rot_mat"] = goal_rot_mat
            self.obj_info["goal_trans_mat"] = goal_trans
        else:
            raise AssertionError('sensor type is not included yet')

        obj_mesh = DotDict(self.obj_info)
        del self.obj_info['verts_sample']
        del self.obj_info['verts_normal']
        if self.sensot_type == 'all_pointnetL_pre':
            signed_dis, dis_min_idx = self.singed_distance_sensors(obj_mesh, joint_sites, ifBatch=True)  # (N, 80)
            ambient_shape = self.ambient_sensors(hand_rot,  # (N, 3, 3)     # (N, 1000)
                                                 hand_pos,   # (N, 3)
                                                 obj_mesh,
                                                 ifBatch=True)  # the root joints of middle finger
            sensors = torch.cat([signed_dis, ambient_shape], dim=-1)   # (N, 80) + (N, 1000) -> (N, 1080)
            pn_feat = self.get_demon_pointnet_feature(obj_mesh.verts_sample, dis_min_idx, ifBatch=True)  # (N, 2304)

            return sensors, pn_feat[:, 1024:]
        elif self.sensot_type in ["surf"]:
            signed_dis, _ = self.singed_distance_sensors(obj_mesh, joint_sites, ifBatch=True)  # (N, 80)
            return signed_dis
        elif self.sensot_type in ["surf2g", "surfv2g", "surfvv2g", "surfh2g"]:
            signed_dis, signed_dis2goal, _ = self.singed_distance_sensors_2gversion(obj_mesh, joint_sites,
                                                                                              ifBatch=True,
                                                                                              sensor_type=self.sensot_type)  # (N, 80)
            return signed_dis, signed_dis2goal
        elif self.sensot_type == 'pointnetG_pre':
            # signed_dis = self.singed_distance_sensors(obj_mesh, joint_sites)
            obj_sampled_verts = obj_mesh.verts_sample

            pn_feat = self.get_demon_pointnet_feature(obj_sampled_verts, ifBatch=True)
            return pn_feat
        elif self.sensot_type in ["dexrep_toGoal", "dexrep_VtoGoal", "dexrep_HtoGoal", "dexrep_VVtoGoal"]:
            signed_dis, signed_dis2goal, dis_min_idx = self.singed_distance_sensors_2gversion(obj_mesh, joint_sites, ifBatch=True,
                                                                                              sensor_type=self.sensot_type)  # (N, 80)
            ambient_shape = self.ambient_sensors(hand_rot,  # (N, 3, 3)     # (N, 1000)
                                                 hand_pos,  # (N, 3)
                                                 obj_mesh,
                                                 ifBatch=True)  # the root joints of middle finger
            sensors = torch.cat([signed_dis, ambient_shape], dim=-1)  # (N, 80) + (N, 1000) -> (N, 1080)
            pn_feat = self.get_demon_pointnet_feature_2(obj_mesh.verts_sample, dis_min_idx, ifBatch=True)  # (N, 2304 - 1024 = 1280)
            # pn_feat_origin = self.get_demon_pointnet_feature(obj_mesh.verts_sample, dis_min_idx, ifBatch=True)
            # pn_feat_origin = pn_feat_origin[:, 1024:]
            # return sensors, pn_feat[:, 1024:], signed_dis2goal
            return sensors, pn_feat, signed_dis2goal
        elif self.sensot_type == "dexrep_double":
            # hand 2 obj
            signed_dis, dis_min_idx = self.singed_distance_sensors(obj_mesh, joint_sites, ifBatch=True)  # (N, 80)
            ambient_shape = self.ambient_sensors(hand_rot,  # (N, 3, 3)     # (N, 1000)
                                                 hand_pos,  # (N, 3)
                                                 obj_mesh,
                                                 ifBatch=True)  # the root joints of middle finger
            sensors = torch.cat([signed_dis, ambient_shape], dim=-1)  # (N, 80) + (N, 1000) -> (N, 1080)
            pn_feat = self.get_demon_pointnet_feature(obj_mesh.verts_sample, dis_min_idx, ifBatch=True)  # (N, 2304)

            # hand 2 goal
            obj_mesh.verts_sample = torch.matmul(torch.clone(obj_verts), goal_rot_mat.transpose(-1, -2)) + goal_trans
            obj_mesh.verts_normal = torch.matmul(torch.clone(obj_norms), goal_rot_mat.transpose(-1, -2))
            obj_mesh.rot_mat = goal_rot_mat
            obj_mesh.trans_mat = goal_trans
            signed_dis_goal, dis_min_idx_goal = self.singed_distance_sensors(obj_mesh, joint_sites, ifBatch=True)  # (N, 80)
            ambient_shape_goal = self.ambient_sensors(hand_rot,  # (N, 3, 3)     # (N, 1000)
                                                 hand_pos,  # (N, 3)
                                                 obj_mesh,
                                                 ifBatch=True)  # the root joints of middle finger
            sensors_goal = torch.cat([signed_dis_goal, ambient_shape_goal], dim=-1)  # (N, 80) + (N, 1000) -> (N, 1080)
            pn_feat_goal = self.get_demon_pointnet_feature(obj_mesh.verts_sample, dis_min_idx_goal, ifBatch=True)  # (N, 2304)

            return sensors, pn_feat[:, 1024:], sensors_goal, pn_feat_goal[:, 1024:]
        else:
            raise AssertionError('sensor type is not included yet')

    def get_demon_pointnet_feature(self, pcd, min_loc_idx=None, ifBatch=False):
        pcd = pc_normalize_tensor(pcd)  # (N, 2048, 3)
        # self.instance(pcd)
        if isinstance(pcd, np.ndarray):
            pcd_tensor = torch.from_numpy(pcd.reshape(-1, 2048, 3)).float()
        else:
            pcd_tensor = torch.clone(pcd).reshape(-1, 2048, 3)


        pcd_tensor = pcd_tensor.transpose(2, 1)      # (N, 3, 2048)
        # split pcd to save GPU mem
        g_feat_res, l_feat_res = [], []
        pcd_tensor_slices = [pcd_tensor[i*self.batch_split_num: (i+1)*self.batch_split_num, ...] \
                             for i in range(int((pcd_tensor.shape[0] + self.batch_split_num - 1) / self.batch_split_num))]
        with torch.no_grad():
            for slice in pcd_tensor_slices:
                g_feat_slice, l_feat_slice = self.PN.pcd_encoder(slice.float()) # (N, 1024), (N, 64, 2048)
                g_feat_res.append(g_feat_slice)
                l_feat_res.append(l_feat_slice)
            g_feat = torch.cat(g_feat_res, dim=0)
            l_feat = torch.cat(l_feat_res, dim=0)

        # print('pcd', time.time() - ts)
        if min_loc_idx is not None:
            env_num = pcd.size(0)
            loc_idx = min_loc_idx    # (N, 20)
            l_feat = self.select_pointnet_local_feat(l_feat, loc_idx)    # (N, 20, 64)
            pn_feat = torch.cat([g_feat, l_feat.reshape(env_num, -1)], dim=-1) if ifBatch else \
                torch.cat([g_feat.view(-1), l_feat.view(-1)])  # (N, 2304)
        else:
            pn_feat = g_feat if ifBatch else g_feat.view(-1)

        return pn_feat

    def get_demon_pointnet_feature_2(self, pcd, min_loc_idx=None, ifBatch=False):
        pcd = pc_normalize_tensor(pcd)  # (N, 2048, 3)
        # self.instance(pcd)
        if isinstance(pcd, np.ndarray):
            pcd_tensor = torch.from_numpy(pcd.reshape(-1, 2048, 3)).float()
        else:
            pcd_tensor = torch.clone(pcd).reshape(-1, 2048, 3)

        pcd_tensor = pcd_tensor.transpose(2, 1)      # (N, 3, 2048)
        # split pcd to save GPU mem
        l_feat_res = []
        pcd_tensor_slices = [pcd_tensor[i*self.batch_split_num: (i+1)*self.batch_split_num, ...] \
                             for i in range(int((pcd_tensor.shape[0] + self.batch_split_num - 1) / self.batch_split_num))]

        env_num = pcd.size(0)
        assert min_loc_idx is not None
        assert ifBatch
        loc_idx_slices = [min_loc_idx[i*self.batch_split_num: (i+1)*self.batch_split_num, ...] \
                          for i in range(int((min_loc_idx.shape[0] + self.batch_split_num - 1) / self.batch_split_num))]
        process_zip = list(zip(pcd_tensor_slices, loc_idx_slices))
        with torch.no_grad():
            for (pcd_slice, loc_slice) in process_zip:
                _, l_feat_slice = self.PN.pcd_encoder(pcd_slice.float()) # (N, 1024), (N, 64, 2048)
                # g_feat_res.append(g_feat_slice)
                l_feat_slice = self.select_pointnet_local_feat(l_feat_slice, loc_slice)
                l_feat_res.append(l_feat_slice)
            # g_feat = torch.cat(g_feat_res, dim=0)
            l_feat = torch.cat(l_feat_res, dim=0)
            del l_feat_res

        pn_feat = l_feat.reshape(env_num, -1)

        return pn_feat

    def select_pointnet_local_feat(self, l_feat, loc_idx):
        if type(loc_idx) is not torch.Tensor:
            loc_idx = torch.LongTensor(loc_idx).to(self.device)
        if loc_idx.ndim == 1:
            loc_idx = loc_idx.unsqueeze(dim=0)
        loc_idx_ = loc_idx.unsqueeze(dim=-1).repeat(1, 1, l_feat.shape[1])
        l_feat = torch.gather(l_feat.transpose(2, 1), 1, loc_idx_)
        return l_feat

    def instance(self, pcd):
        # pcd = pc_normalize_tensor(pcd)
        if isinstance(pcd, np.ndarray):
            pcd_tensor = torch.from_numpy(pcd.reshape(-1, 2048, 3)).float().to(self.device)
        else:
            pcd_tensor = torch.clone(pcd).reshape(-1, 2048, 3)

        pcd_tensor = pcd_tensor.transpose(2, 1)
        pcd_pre = self.PN(pcd_tensor)['pcd_rec'].cpu().numpy().squeeze()
        pcd_gt = pcd
        import open3d as o3d
        pcd_ori = o3d.geometry.PointCloud()
        pcd_ori.points = o3d.utility.Vector3dVector(pcd_gt.cpu().numpy())
        pcd_ori.paint_uniform_color([0, 0, 1])
        # cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        pcd_scale = o3d.geometry.PointCloud()
        pcd_scale.points = o3d.utility.Vector3dVector(pcd_pre)
        pcd_scale.paint_uniform_color([0, 1, 0])
        cords = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        o3d.visualization.draw_geometries([pcd_ori, pcd_scale, cords])

    def singed_distance_sensors_2gversion(self, object, joints_pos, ifBatch=True, sensor_type="dexrep_toGoal",
                                          batch_split=8000):
        assert ifBatch == True
        batch_size_N = joints_pos.shape[0]

        if sensor_type in ["dexrep_toGoal", "dexrep_VtoGoal", "dexrep_VVtoGoal", "surf2g", "surfv2g", "surfvv2g"]:
            # prepare
            finger_points = joints_pos  # (N, 20, 3)
            origin_pcb = object.verts_sample  # (N, 2048, 3)
            origin_pcb_normal = object.verts_normal  # (N, 2048, 3)
            obj_rot_mat = object.rot_mat  # (N, 3, 3)
            obj_trans = object.trans_mat  # (N, 1, 3)
            obj_pcb = torch.matmul(origin_pcb, obj_rot_mat.transpose(-1, -2)) + obj_trans  # (N, 2048, 3)
            goal_rot_mat = object.goal_rot_mat  # (N, 3, 3)
            goal_trans = object.goal_trans_mat  # (N, 1, 3)

            # hand_joints 2 obj, split batch to save memory
            # dis = torch.cdist(finger_points, obj_pcb)  # (N, 20, 2048)
            # dis_min, dis_min_idx = torch.min(dis, dim=-1)
            dis_min, dis_min_idx = util.split_torch_dist(finger_points, obj_pcb, split_batch=batch_split)

            self.joints2objsurface = dis_min[:, 1:].cpu().numpy().copy()   # (N, 19)
            dis_min = dis_min.clip(max=self.dis_max_range)  # (N, 20)

            # origin points
            origin_min_points = origin_pcb[torch.arange(origin_pcb.size(0)).unsqueeze(1), dis_min_idx, :]  # (N, 20, 3)
            origin_min_norms = origin_pcb_normal[torch.arange(origin_pcb_normal.size(0)).unsqueeze(1), dis_min_idx, :]  # (N, 20, 3)
            # obj surf normals
            con_verts_min_normal = torch.matmul(torch.clone(origin_min_norms), obj_rot_mat.transpose(-1, -2))
            obj_mindis_points = obj_pcb[torch.arange(obj_pcb.size(0)).unsqueeze(1), dis_min_idx, :]  # (N, 20, 3)
            if sensor_type in ["dexrep_toGoal", "dexrep_VtoGoal", "surf2g", "surfv2g"]:
                SD = torch.cat([con_verts_min_normal.reshape(batch_size_N, -1), dis_min * 10],  # (N, 80)
                               dim=-1)  # (N, 20, 3) + (N, 20) -> (N, 80)
            elif sensor_type in ["dexrep_VVtoGoal", "surfvv2g"]:
                # distance Vector
                hand2obj_vector = obj_mindis_points - finger_points
                SD = torch.cat([con_verts_min_normal.reshape(batch_size_N, -1), hand2obj_vector.reshape(batch_size_N, -1) * 10],
                               dim=-1)
            else:
                raise NotImplementedError(f"sensor type {sensor_type} not impleted!")

            # obj 2 goal distance
            goal_mindis_points = torch.matmul(origin_min_points, goal_rot_mat.transpose(-1, -2)) + goal_trans
            if sensor_type in ["dexrep_toGoal", "surf2g"]:
                squared_diff = (goal_mindis_points - obj_mindis_points) ** 2
                obj2goal = torch.sqrt(torch.sum(squared_diff, dim=2))
                obj2goal = obj2goal.clip(max=self.dis_max_range)  # (N, 20)
                goal_mindis_norms = torch.matmul(torch.clone(origin_min_norms), goal_rot_mat.transpose(-1, -2))  # (N, 20, 3)
                SD2g = torch.cat([goal_mindis_norms.reshape(batch_size_N, -1), obj2goal * 10],
                                 dim=-1)  # (N, 20, 3) + (N, 20) -> (N, 80)
            elif sensor_type in ["dexrep_VtoGoal", "dexrep_VVtoGoal", "surfv2g", "surfvv2g"]:
                obj2goal_vector = goal_mindis_points - obj_mindis_points  # (N, 20, 3)
                goal_mindis_norms = torch.matmul(origin_min_norms, goal_rot_mat.transpose(-1, -2))  # (N, 20, 3)
                obj2goal_vector = obj2goal_vector.clip(max=self.dis_max_range)
                SD2g = torch.cat(
                    [goal_mindis_norms.reshape(batch_size_N, -1), obj2goal_vector.reshape(batch_size_N, -1) * 10],
                    dim=-1
                )
            else:
                raise NotImplementedError(f"sensor type {sensor_type} not impleted!")

            # replace object
            object.verts_sample = obj_pcb
            object.verts_normal = torch.matmul(origin_pcb_normal, obj_rot_mat.transpose(-1, -2))
        elif sensor_type in ["dexrep_HtoGoal", "surfh2g"]:
            # prepare
            finger_points = joints_pos  # (N, 20, 3)
            origin_pcb = object.verts_sample  # (N, 2048, 3)
            origin_pcb_normal = object.verts_normal  # (N, 2048, 3)
            obj_rot_mat = object.rot_mat  # (N, 3, 3)
            obj_trans = object.trans_mat  # (N, 1, 3)
            goal_rot_mat = object.goal_rot_mat  # (N, 3, 3)
            goal_trans = object.goal_trans_mat  # (N, 1, 3)
            goal_pcb = torch.matmul(torch.clone(origin_pcb), goal_rot_mat.transpose(-1, -2)) + goal_trans  # (N, 2048, 3)
            goal_pcb_normal = torch.matmul(torch.clone(origin_pcb_normal), goal_rot_mat.transpose(-1, -2)) # (N, 2048, 3)

            # hand_joints 2 goal
            # dis2g = torch.cdist(finger_points, goal_pcb)  # (N, 20, 2048)
            # dis2g_min, dis2g_min_idx = torch.min(dis2g, dim=-1)
            dis2g_min, dis2g_min_idx = util.split_torch_dist(finger_points, goal_pcb, split_batch=batch_split)
            dis2g_min = dis2g_min.clip(max=self.dis_max_range)  # (N, 20)
            # goal points normal
            goal_mindis_norms = torch.clone(
                goal_pcb_normal[torch.arange(goal_pcb_normal.size(0)).unsqueeze(1), dis2g_min_idx, :])
            SD2g = torch.cat([goal_mindis_norms.reshape(batch_size_N, -1), dis2g_min * 10],  # (N, 80)
                             dim=-1)

            # prepare
            del goal_pcb
            del goal_pcb_normal
            obj_pcb = torch.matmul(origin_pcb, obj_rot_mat.transpose(-1, -2)) + obj_trans  # (N, 2048, 3)

            # hand_joints 2 obj
            # dis = torch.cdist(finger_points, obj_pcb)  # (N, 20, 2048)
            # dis_min, dis_min_idx = torch.min(dis, dim=-1)
            dis_min, dis_min_idx = util.split_torch_dist(finger_points, obj_pcb, split_batch=batch_split)

            self.joints2objsurface = dis_min[:, 1:].cpu().numpy().copy()  # (N, 19)
            dis_min = dis_min.clip(max=self.dis_max_range)  # (N, 20)
            # origin points
            origin_min_norms = origin_pcb_normal[torch.arange(origin_pcb_normal.size(0)).unsqueeze(1), dis_min_idx, :]  # (N, 20, 3)
            # obj surf normals
            con_verts_min_normal = torch.matmul(origin_min_norms, obj_rot_mat.transpose(-1, -2))
            SD = torch.cat([con_verts_min_normal.reshape(batch_size_N, -1), dis_min * 10],  # (N, 80)
                           dim=-1)  # (N, 20, 3) + (N, 20) -> (N, 80)

            # replace object
            object.verts_sample = obj_pcb
            object.verts_normal = torch.matmul(origin_pcb_normal, obj_rot_mat.transpose(-1, -2))
        else:
            raise NotImplementedError(f"sensor type {sensor_type} not impleted!")

        return SD, SD2g, dis_min_idx

    def singed_distance_sensors(self, object, joints_pos, ifBatch=False, batch_split=8000):
        finger_points = joints_pos   # (N, 20, 3)

        # if self.obj_info['verts_sample_id'] is not None:
        #     obj_pcb = np.asarray(object.vertices)[self.obj_info['verts_sample_id']]
        #     obj_pcb_normal = np.asarray(object.vertex_normals)[self.obj_info['verts_sample_id']]
        # else:
        #     obj_pcb = np.asarray(object.points)
        #     obj_pcb_normal = np.asarray(object.normals)
        obj_pcb = object.verts_sample           # (N, 2048, 3)
        obj_pcb_normal = object.verts_normal    # (N, 2048, 3)

        # dis = torch.cdist(finger_points, obj_pcb)      # (N, 20, 2048)
        # dis_min, dis_min_idx = torch.min(dis, dim=-1)  # (N, 20), (N, 20)        # dis_min_idx = torch.argmin(dis, dim=-1)
        dis_min, dis_min_idx = util.split_torch_dist(finger_points, obj_pcb, split_batch=batch_split)
        # dis = cdist(finger_points, obj_pcb, metric='euclidean')
        # dis_min = np.min(dis, axis=-1)
        # dis_min_idx = torch.argmin(dis, dim=-1)

        if ifBatch:
            self.joints2objsurface = dis_min[:, 1:].cpu().numpy().copy()   # (N, 19)
        else:
            self.joints2objsurface = dis_min[1:].cpu().numpy().copy()

        dis_min = dis_min.clip(max=self.dis_max_range)          # (N, 20)

        # con_verts_min_normal_deonm = np.linalg.norm(con_verts_min_normal, axis=-1, keepdims=True)
        # con_verts_min_normal = con_verts_min_normal / con_verts_min_normal_deonm
        if ifBatch:
            con_verts_min_normal = obj_pcb_normal[torch.arange(obj_pcb_normal.size(0)).unsqueeze(1), dis_min_idx, :]   # (N, 20, 3)
            batch_size_N = joints_pos.shape[0]
            SD = torch.cat([con_verts_min_normal.reshape(batch_size_N, -1), dis_min * 10],  # (N, 80)
                           dim=-1)  # (N, 20, 3) + (N, 20) -> (N, 80)
        else:
            con_verts_min_normal = obj_pcb_normal[dis_min_idx]
            SD = torch.cat([con_verts_min_normal.flatten(), dis_min * 10])

        # SD = np.concatenate([dis_min.reshape(-1, 1), con_verts_min_normal], axis=1)
        return SD, dis_min_idx

    def ambient_sensors(self, rotation, trans, obj, ifBatch=False):
        # cube = trimesh.primitives.Box(extents=(0.02, 0.02, 0.02))
        # cube.show()
        cords = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

        # cubes = cubes_VG.v_grid
        # cubes_VG.show(hand)
        # cubes_VG.translate([0.1, 0.14, 0.08])
        # cubes_VG.show([hand, cords])
        if ifBatch:
            env_name = rotation.shape[0]
            all_cube_center = self.create_grid(self.length, self.length_per_cube,
                                               dtype=torch.float32).repeat(env_name, 1, 1).to(self.device)
            off_set = torch.tensor(np.array([0, 0.045, 0]), dtype=torch.float32).to(self.device)  # (3)
        else:
            all_cube_center = self.create_grid(self.length, self.length_per_cube,
                                               dtype=torch.float32).to(self.device)  # (1000, 3)
            off_set = torch.tensor(np.array([0, 0.045, 0]), dtype=torch.float32).to(self.device)  # (3)
        all_cube_center += off_set                            # (1000, 3)

        all_cube_center = torch.matmul(all_cube_center, rotation.transpose(-1, -2)) + trans.unsqueeze(-2)  # (1000, 3) * (N, 3, 3) + (N, 1, 3) -> (N, 1000, 3)

        occupied_area = torch.zeros(all_cube_center.shape[:-1]).float().to(self.device)  # (N, 1000)
        # ts = time.time()
        # dis = scene.compute_signed_distance(all_cube_center.astype(np.float32)).numpy()
        # obj_pcb = o3d.geometry.PointCloud()
        # obj_pcb.points = o3d.utility.Vector3dVector(np.asarray(obj.vertices)[self.obj_info['verts_sample_id']])
        # c_pcb = o3d.geometry.PointCloud()
        # c_pcb.points = o3d.utility.Vector3dVector(all_cube_center)
        # dis = np.asarray(c_pcb.compute_point_cloud_distance(obj_pcb))
        # print('test %f '%(time.time()-ts))
        o_xyz = obj.verts_sample    # (N, 2048, 3)

        # split all_cube_center to save gpu mem
        dis_min_res = []
        all_cube_center_slices = []
        o_xyz_slices = []
        for i in range(int((all_cube_center.shape[0] + self.batch_split_num - 1) / self.batch_split_num)):
            all_cube_center_slices.append(all_cube_center[i * self.batch_split_num: (i + 1) * self.batch_split_num, ...])
            o_xyz_slices.append(o_xyz[i * self.batch_split_num: (i + 1) * self.batch_split_num, ...])
        for cube_center_slice, o_xyz_slice in zip(all_cube_center_slices, o_xyz_slices):
            dis_slice = torch.cdist(cube_center_slice, o_xyz_slice)  # (N, 1000, 2048)，计算Cube上1000个中心距离obj每个点的距离
            dis_min_res.append(
                dis_slice.min(dim=-1)[0]  # (N, 1000)
            )
            del dis_slice  # free gpu mem
        dis_min = torch.cat(dis_min_res, dim=0)
        # dis = torch.cdist(all_cube_center, o_xyz)  # (1000, 2048)
        # dis_min = dis.min(dim=1)[0]
        #
        is_inside = dis_min < self.length_per_cube / 2  # (N, 1000) type=bool
        occupied_area[is_inside] = 1                    # (N, 1000)
        # rays = np.concatenate([center_per_cube, obj_center-center_per_cube], axis=1)
        #
        # rays = rays.astype(np.float32)
        #
        # # hit_info = scene.cast_rays(rays)
        # # hit_dis = hit_info['t_hit'].numpy().clip(max=0.2)
        # occupied_area = np.zeros(center_per_cube.shape[0])
        # intersection_counts = scene.count_intersections(rays).numpy()
        # is_inside = intersection_counts % 2 == 1
        # occupied_area[is_inside] = 1
        # print(occupied_area)
        return occupied_area

    def get_created_cube_center(self, batch_num):
        if self.all_cubes_center is None:
            self.cube_center_batch_num = batch_num
            all_cube_center = self.create_grid(self.length, self.length_per_cube,
                                               dtype=torch.float32).repeat(batch_num, 1, 1).to(self.device)
            off_set = torch.tensor(np.array([0, 0.045, 0]), dtype=torch.float32).to(self.device)
            all_cube_center += off_set
            self.all_cubes_center = all_cube_center
        else:
            # batch_num not match, recompute
            if batch_num != self.cube_center_batch_num:
                self.cube_center_batch_num = batch_num
                all_cube_center = self.create_grid(self.length, self.length_per_cube,
                                                   dtype=torch.float32).repeat(batch_num, 1, 1).to(self.device)
                off_set = torch.tensor(np.array([0, 0.045, 0]), dtype=torch.float32).to(self.device)
                all_cube_center += off_set
                self.all_cubes_center = all_cube_center

    # def ambient_sdf_sensors(self, rotation, trans, obj, sdfs, ifBatch=False):
    #     if ifBatch:
    #         self.get_created_cube_center(rotation.shape[0])
    #         occupied_area = torch.zeros(self.all_cubes_center.shape[:-1]).float().to(self.device)  # (N, 1000)
    #         # 所有环境一起查询
    #         R_hand = rotation.transpose(-1, -2)
    #         R_obj_inv = torch.inverse(obj.rot_mat.transpose(-1, -2))
    #         T_hand, T_obj = trans.unsqueeze(-2), obj.trans_mat
    #         R_mat = torch.matmul(R_hand, R_obj_inv)
    #         T_mat = torch.matmul((T_hand - T_obj), R_obj_inv)
    #         all_cube_center = torch.matmul(self.all_cubes_center, R_mat) + T_mat
    #         all_query_points = all_cube_center
    #         for i in range(rotation.shape[0]):
    #             obj_sdf = sdfs[obj.mesh_name[i]]
    #             query_result, outside_mask = sdf_signs_query_points(all_query_points[i], **obj_sdf)
    #             is_inside = query_result < self.length_per_cube / 2
    #             occupied_area[i][is_inside] = 1
    #             occupied_area[i][outside_mask] = 0
    #         # # 每个环境单独查询
    #         # for i in range(rotation.shape[0]):
    #         #     # rotate cube_center: (Cube_Center * R_hand + T_hand - T_obj) * R_obj^-1 = Cube_Center * R_hand * R_obj^-1 + (T_hand - T_obj) * R_obj^-1
    #         #     R_hand = rotation[i].transpose(-1, -2)
    #         #     R_obj_inv = torch.inverse(obj.rot_mat[i].transpose(-1, -2))
    #         #     T_hand, T_obj = trans.unsqueeze(-2)[i], obj.trans_mat[i]
    #         #     R_mat = torch.matmul(R_hand, R_obj_inv)
    #         #     T_mat = torch.matmul((T_hand - T_obj), R_obj_inv)
    #         #     cube_center = torch.matmul(self.all_cubes_center[i], R_mat) + T_mat
    #         #     # cube_center = all_cube_center[i] - obj.trans_mat[i]
    #         #     # cube_center = torch.matmul(cube_center, torch.inverse(obj.rot_mat[i].transpose(-1, -2)))
    #         #     # convert query points type
    #         #     query_points = cube_center.to(self.device)
    #         #     # distance query
    #         #     obj_sdf = sdfs[obj.mesh_name[i]]
    #         #     query_result, outside_mask = sdf_signs_query_points(query_points, **obj_sdf)
    #         #     is_inside = query_result < self.length_per_cube / 2
    #         #     occupied_area[i][is_inside] = 1
    #         #     occupied_area[i][outside_mask] = 0
    #     else:
    #         raise KeyError("Not impleted!")
    #     return occupied_area


    # def ambient_sdf_sensors_o3d(self, rotation, trans, obj, sdfs, ifBatch=False):
    #     if ifBatch:
    #         self.get_created_cube_center(rotation.shape[0])
    #         occupied_area = torch.zeros(self.all_cubes_center.shape[:-1]).float().to(self.device)  # (N, 1000)
    #
    #         # all_cube_center = torch.matmul(all_cube_center, rotation.transpose(-1, -2)) + trans.unsqueeze(
    #         #     -2)  # (1000, 3) * (N, 3, 3) + (N, 1, 3) -> (N, 1000, 3)
    #         # 每个环境单独查询
    #         for i in range(rotation.shape[0]):
    #             # rotate cube_center: (Cube_Center * R_hand + T_hand - T_obj) * R_obj^-1 = Cube_Center * R_hand * R_obj^-1 + (T_hand - T_obj) * R_obj^-1
    #             R_hand = rotation.transpose(-1, -2)[i]
    #             R_obj_inv = torch.inverse(obj.rot_mat[i].transpose(-1, -2))
    #             T_hand, T_obj = trans.unsqueeze(-2)[i], obj.trans_mat[i]
    #             R_mat = torch.matmul(R_hand, R_obj_inv).cpu().numpy()
    #             T_mat = torch.matmul((T_hand - T_obj), R_obj_inv).cpu().numpy()
    #             cube_center = np.matmul(self.all_cubes_center[i], R_mat) + T_mat
    #             # cube_center = all_cube_center[i] - obj.trans_mat[i]
    #             # cube_center = torch.matmul(cube_center, torch.inverse(obj.rot_mat[i].transpose(-1, -2)))
    #             # convert query points type
    #             query_points = o3d.core.Tensor(cube_center, dtype=o3d.core.Dtype.Float32)
    #             # distance query
    #             sdf = sdfs[obj.mesh_name[i]]
    #             signed_distance = sdf.compute_signed_distance(query_points).numpy()
    #             occupied_area[i][signed_distance < 0] = 1
    #     else:
    #         raise KeyError("Not impleted!")
    #     return occupied_area