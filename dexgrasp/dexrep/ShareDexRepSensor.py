

import os
import numpy as np
import trimesh
import torch
import point_cloud_utils as pcu
import open3d as o3d


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def points2pcd(points, normals=None, colors=None):
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        obj_pcd.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        obj_pcd.colors = o3d.utility.Vector3dVector(colors)

    return obj_pcd


def load_stl_files(dir_list, sample_params=None):
    load_meshes = {}
    sampled_pcds = {}

    for dir in dir_list:
        stl_files = os.listdir(dir)
        for stl_f in stl_files:
            obj_mesh = trimesh.load(dir + stl_f)
            load_meshes[stl_f[:-4]] = obj_mesh
            if sample_params is not None:
                if sample_params["method"] == "average":
                    # sample points
                    object_verts = np.array(obj_mesh.vertices)
                    object_normals = np.array(obj_mesh.vertex_normals)
                    object_faces = np.array(obj_mesh.faces)

                    fid, bc = pcu.sample_mesh_random(object_verts, object_faces, sample_params["num_points"])
                    sampled_points = pcu.interpolate_barycentric_coords(object_faces, fid, bc, object_verts)
                    sampled_normals = pcu.interpolate_barycentric_coords(object_faces, fid, bc, object_normals)
                    # create ply
                    obj_pcd = points2pcd(sampled_points, normals=sampled_normals)
                    # result
                    sampled_pcds[stl_f[:-4]] = obj_pcd
                elif sample_params["method"] == "verts":
                    object_vert = np.array(obj_mesh.vertices)
                    object_normals = np.array(obj_mesh.vertex_normals)

                    selected = sample_params["rand"].randint(
                          low=0, high=object_vert.shape[0], size=sample_params["num_points"])
                    sampled_points = object_vert[selected].copy()
                    sampled_normals = object_normals[selected].copy()
                    # create ply
                    obj_pcd = points2pcd(sampled_points, normals=sampled_normals)
                    # result
                    sampled_pcds[stl_f[:-4]] = obj_pcd
                else:
                    sample_method = sample_params["method"]
                    raise KeyError(f"sample method {sample_method} not impleted!")
            else:
                raise KeyError("no sample not impleted!")


    return load_meshes, sampled_pcds


class SharedDexRepSensor:
    def __init__(self, args, device):
        import dexrep
        self.Sensor = dexrep.DexRep(args, device=device)
        # self.task_name = args["task_name"]

        self.scaled_sampled_points = {}
        self.scaled_sampled_normals = {}

        self.obs_batch_obj_points = []
        self.obs_batch_obj_normals = []

        if "dexrep" in args.keys():
            self.sample_method = args["dexrep"]["sample_method"]
            self.sample_num_points = args["dexrep"]["sample_num_points"]
            self.BatchNormPnFeat = args["dexrep"]["batch_norm_pnfeat"]
        elif "pnG" in args.keys():
            self.sample_method = args["pnG"]["sample_method"]
            self.sample_num_points = args["pnG"]["sample_num_points"]
        else:
            raise KeyError(f"sample_method not impletement")
        # self.rand = RandomState(args.seed)

        # self.meshes, self.sampled_pcds = load_stl_files(stl_files_dir_list, sample_params=sample_params)
        # # Generate SDF
        # self.mesh_sdfs = generate_sdf_mine(self.meshes)

    def load_cache_stl_file(self, obj_path, obj_idx, scale=1.0):
        obj_mesh = trimesh.load(str(obj_path))
        if isinstance(obj_mesh, trimesh.Scene):
            obj_mesh = trimesh.util.concatenate([geometry for geometry in obj_mesh.geometry.values()])
        # scale mesh
        obj_mesh = obj_mesh.apply_scale(scale)
        # sample points and normals
        if self.sample_method == "average":
            object_verts = np.array(obj_mesh.vertices)
            object_normals = np.array(obj_mesh.vertex_normals)
            object_faces = np.array(obj_mesh.faces)
            fid, bc = pcu.sample_mesh_random(object_verts, object_faces, self.sample_num_points)
            sampled_points = pcu.interpolate_barycentric_coords(object_faces, fid, bc, object_verts)
            sampled_normals = pcu.interpolate_barycentric_coords(object_faces, fid, bc, object_normals)
        else:
            raise KeyError(f"sample_method <{self.sample_method}> not impleted!")
        # add to pcd list
        self.scaled_sampled_points[obj_idx] = sampled_points
        self.scaled_sampled_normals[obj_idx] = sampled_normals

    def load_batch_env_obj(self, env_obj_idx):
        # load env points and normals
        self.obs_batch_obj_points.append(
            np.copy(self.scaled_sampled_points[env_obj_idx])
        )
        self.obs_batch_obj_normals.append(
            np.copy(self.scaled_sampled_normals[env_obj_idx])
        )


    def get_perception_data(self,
                     body_xpose,
                     body_xquat,
                     joints_site,
                     sampled_points,
                     sampled_normals
                     ):
        dexrep_feat =self.Sensor.get_perception_data(
            body_xpose,
            body_xquat,
            joints_site,
            sampled_points,
            sampled_normals,
            # sdfs=self.mesh_sdfs
        )
        dexrep_feat = np.concatenate(dexrep_feat)
        return dexrep_feat.flatten()
        # return dexrep_feat[0], dexrep_feat[1]

    def get_batch_perception_data(self,
                     body_xpose,
                     body_xquat,
                     joints_site,
                     hand_pos,
                     hand_rot,
                     sampled_points,
                     sampled_normals,
                     clip_range,
                     mesh_names=None
                     )->torch.Tensor:
        (sensors, pn_feat) = self.Sensor.get_batch_perception_data(
            body_xpose,
            body_xquat,
            joints_site,
            hand_pos,
            hand_rot,
            sampled_points,
            sampled_normals,
            # mesh_names=mesh_names,
            # sdfs=self.mesh_sdfs
        )
        # set nan to zero
        # sensors = self.detect_nan_setZero(sensors)
        # pn_feat = self.detect_nan_setZero(pn_feat)
        # batch normalize pn_feat
        if self.BatchNormPnFeat:
            pn_feat_mean = pn_feat.mean(axis=0, keepdim=True)
            pn_feat_std = pn_feat.std(dim=0, keepdim=True)
            pn_feat = (pn_feat - pn_feat_mean) / (pn_feat_std + 1e-8)

            pn_feat = torch.clamp(pn_feat, -clip_range, clip_range)
        # clip sensors
        sensors = torch.clamp(sensors, -clip_range, clip_range)

        dexrep_feat = torch.cat([sensors, pn_feat], dim=-1)
        return dexrep_feat

    @staticmethod
    def detect_nan_setZero(input_tensor, value=0):
        nan_exists = torch.isnan(input_tensor)
        if nan_exists.any():
            nan_loc = torch.where(nan_exists)
            nan_env = torch.unique(nan_loc[0])
            # set this env to zero
            for env_idx in nan_env:
                if value == 0:
                    input_tensor[env_idx] = torch.zeros((input_tensor.shape[1]))
                else:
                    input_tensor[env_idx] = torch.ones((input_tensor.shape[1])) * value
        return input_tensor

    def pre_observation(self, obj_pos, obj_rot, hand_pos, hand_rot, joints_sate, clip_range):
        # batch 操作
        # # PreProcess
        # body_xpose_list = []
        # body_xquat_list = []
        # joints_site_list = []
        # sampled_points_list = []
        # sampled_normals_list = []
        # mesh_names_list = []
        # for env_ob in :
        #     body_xpose_list.append(env_ob["body_xpose"])
        #     body_xquat_list.append(env_ob["body_xquat"])
        #     joints_site_list.append(env_ob["joints_site"])
        if isinstance(self.obs_batch_obj_points, np.ndarray) or isinstance(self.obs_batch_obj_points, list):
            # self.obs_batch_obj_points = torch.tensor(np.array(self.obs_batch_obj_points), dtype=torch.float32).to(device)
            # self.obs_batch_obj_normals = torch.tensor(np.array(self.obs_batch_obj_normals), dtype=torch.float32).to(device)
            self.obs_batch_obj_points = torch.tensor(np.array(self.obs_batch_obj_points), dtype=torch.float32).to(obj_pos.device)
            self.obs_batch_obj_normals = torch.tensor(np.array(self.obs_batch_obj_normals), dtype=torch.float32).to(obj_pos.device)

        # st2 = time.time()
        dexrep_feat_cuda = self.get_batch_perception_data(
            body_xpose=obj_pos,
            body_xquat=obj_rot,
            joints_site=joints_sate,
            hand_pos=hand_pos,
            hand_rot=hand_rot,
            sampled_points=self.obs_batch_obj_points,
            sampled_normals=self.obs_batch_obj_normals,
            clip_range=clip_range
        )
        dexrep_feat_cuda = dexrep_feat_cuda.float() # 注意，这边double -> float

        return dexrep_feat_cuda

class SharedPnGSensor(SharedDexRepSensor):
    def __init__(self, args):
        self.Sensor.sensot_type = "pointnetG_pre"
        super(SharedPnGSensor, self).__init__(args)

    def get_batch_perception_data(self,
                     body_xpose,
                     body_xquat,
                     joints_site,
                     hand_pos,
                     hand_rot,
                     sampled_points,
                     sampled_normals,
                     clip_range,
                     mesh_names=None
                     )->torch.Tensor:
        pn_feat = self.Sensor.get_batch_perception_data(
            body_xpose,
            body_xquat,
            joints_site,
            hand_pos,
            hand_rot,
            sampled_points,
            sampled_normals,
            # mesh_names=mesh_names,
            # sdfs=self.mesh_sdfs
        )
        # batch normalize pn_feat
        # pn_feat_mean = pn_feat.mean(axis=0, keepdim=True)
        # pn_feat_std = pn_feat.std(dim=0, keepdim=True)
        # pn_feat = (pn_feat - pn_feat_mean) / (pn_feat_std + 1e-8)
        return pn_feat

class SharedDexrep2GSensor(SharedDexRepSensor):
    def __init__(self, args):
        super(SharedDexrep2GSensor, self).__init__(args)
        self.Sensor.sensot_type = "dexrep_toGoal"

    def pre_observation(self, obj_pos, obj_rot, goal_pos, goal_rot,
                        hand_pos, hand_rot, joints_sate, clip_range):
        if isinstance(self.obs_batch_obj_points, np.ndarray) or isinstance(self.obs_batch_obj_points, list):
            self.obs_batch_obj_points = torch.tensor(np.array(self.obs_batch_obj_points), dtype=torch.float32).to(
                device)
            self.obs_batch_obj_normals = torch.tensor(np.array(self.obs_batch_obj_normals), dtype=torch.float32).to(
                device)

        dexrep_feat_cuda = self.get_batch_perception_data(
            body_xpose=obj_pos,
            body_xquat=obj_rot,
            joints_site=joints_sate,
            hand_pos=hand_pos,
            hand_rot=hand_rot,
            sampled_points=self.obs_batch_obj_points,
            sampled_normals=self.obs_batch_obj_normals,
            clip_range=clip_range,
            goal_pos=goal_pos,
            goal_rot=goal_rot
        )
        dexrep_feat_cuda = dexrep_feat_cuda.float()  # 注意，这边double -> float

        return dexrep_feat_cuda

    def get_batch_perception_data(self,
                                  body_xpose,
                                  body_xquat,
                                  joints_site,
                                  hand_pos,
                                  hand_rot,
                                  sampled_points,
                                  sampled_normals,
                                  clip_range,
                                  goal_pos,
                                  goal_rot,
                                  mesh_names=None) ->torch.Tensor:
        sensors, pn_feat, sensors2g = self.Sensor.get_batch_perception_data(
            body_xpose,
            body_xquat,
            joints_site,
            hand_pos,
            hand_rot,
            sampled_points,
            sampled_normals,
            goal_pos=goal_pos,
            goal_quat=goal_rot
        )
        # clip sensors
        sensors = torch.clamp(sensors, -clip_range, clip_range)
        sensors2g = torch.clamp(sensors2g, -clip_range, clip_range)
        dexrep_feat = torch.cat([sensors, pn_feat, sensors2g], dim=-1)
        return dexrep_feat

class SharedSurfSensor(SharedDexRepSensor):
    def __init__(self, args):
        import dexrep
        self.Sensor = dexrep.DexRep(sensor_type="surf")
        self.task_name = args["task_name"]

        self.scaled_sampled_points = {}
        self.scaled_sampled_normals = {}

        self.obs_batch_obj_points = []
        self.obs_batch_obj_normals = []

        if "dexrep" in args.keys():
            self.sample_method = args["dexrep"]["sample_method"]
            self.sample_num_points = args["dexrep"]["sample_num_points"]
            self.BatchNormPnFeat = args["dexrep"]["batch_norm_pnfeat"]
        else:
            raise KeyError(f"sample_method not impletement")


    def get_batch_perception_data(self,
                                  body_xpose,
                                  body_xquat,
                                  joints_site,
                                  hand_pos,
                                  hand_rot,
                                  sampled_points,
                                  sampled_normals,
                                  clip_range,
                                  mesh_names=None) -> torch.Tensor:
        sensors = self.Sensor.get_batch_perception_data(
            body_xpose,
            body_xquat,
            joints_site,
            hand_pos,
            hand_rot,
            sampled_points,
            sampled_normals,
        )
        # clip sensors
        sensors = torch.clamp(sensors, -clip_range, clip_range)
        dexrep_feat = torch.cat([sensors, ], dim=-1)
        return dexrep_feat


class SharedSurf2GSensor(SharedDexRepSensor):
    def __init__(self, args):
        import dexrep
        self.Sensor = dexrep.DexRep(sensor_type="surf2g")
        self.task_name = args["task_name"]

        self.scaled_sampled_points = {}
        self.scaled_sampled_normals = {}

        self.obs_batch_obj_points = []
        self.obs_batch_obj_normals = []

        if "dexrep" in args.keys():
            self.sample_method = args["dexrep"]["sample_method"]
            self.sample_num_points = args["dexrep"]["sample_num_points"]
            self.BatchNormPnFeat = args["dexrep"]["batch_norm_pnfeat"]
        else:
            raise KeyError(f"sample_method not impletement")

    def pre_observation(self, obj_pos, obj_rot, goal_pos, goal_rot,
                        hand_pos, hand_rot, joints_sate, clip_range):
        if isinstance(self.obs_batch_obj_points, np.ndarray) or isinstance(self.obs_batch_obj_points, list):
            self.obs_batch_obj_points = torch.tensor(np.array(self.obs_batch_obj_points), dtype=torch.float32).to(
                device)
            self.obs_batch_obj_normals = torch.tensor(np.array(self.obs_batch_obj_normals), dtype=torch.float32).to(
                device)

        dexrep_feat_cuda = self.get_batch_perception_data(
            body_xpose=obj_pos,
            body_xquat=obj_rot,
            joints_site=joints_sate,
            hand_pos=hand_pos,
            hand_rot=hand_rot,
            sampled_points=self.obs_batch_obj_points,
            sampled_normals=self.obs_batch_obj_normals,
            clip_range=clip_range,
            goal_pos=goal_pos,
            goal_rot=goal_rot
        )
        dexrep_feat_cuda = dexrep_feat_cuda.float()  # 注意，这边double -> float

        return dexrep_feat_cuda


    def get_batch_perception_data(self,
                                  body_xpose,
                                  body_xquat,
                                  joints_site,
                                  hand_pos,
                                  hand_rot,
                                  sampled_points,
                                  sampled_normals,
                                  clip_range,
                                  goal_pos,
                                  goal_rot,
                                  mesh_names=None) -> torch.Tensor:
        sensors, sensors2g = self.Sensor.get_batch_perception_data(
            body_xpose,
            body_xquat,
            joints_site,
            hand_pos,
            hand_rot,
            sampled_points,
            sampled_normals,
            goal_pos=goal_pos,
            goal_quat=goal_rot
        )
        # clip sensors
        sensors = torch.clamp(sensors, -clip_range, clip_range)
        sensors2g = torch.clamp(sensors2g, -clip_range, clip_range)
        dexrep_feat = torch.cat([sensors, sensors2g], dim=-1)
        return dexrep_feat

class SharedSurfV2GSensor(SharedSurf2GSensor):
    def __init__(self, args):
        import dexrep
        self.Sensor = dexrep.DexRep(sensor_type="surfv2g")
        self.task_name = args["task_name"]

        self.scaled_sampled_points = {}
        self.scaled_sampled_normals = {}

        self.obs_batch_obj_points = []
        self.obs_batch_obj_normals = []

        if "dexrep" in args.keys():
            self.sample_method = args["dexrep"]["sample_method"]
            self.sample_num_points = args["dexrep"]["sample_num_points"]
            self.BatchNormPnFeat = args["dexrep"]["batch_norm_pnfeat"]
        else:
            raise KeyError(f"sample_method not impletement")

    def get_batch_perception_data(self,
                                  body_xpose,
                                  body_xquat,
                                  joints_site,
                                  hand_pos,
                                  hand_rot,
                                  sampled_points,
                                  sampled_normals,
                                  clip_range,
                                  goal_pos,
                                  goal_rot,
                                  mesh_names=None) -> torch.Tensor:
        sensors, sensors2g = self.Sensor.get_batch_perception_data(
            body_xpose,
            body_xquat,
            joints_site,
            hand_pos,
            hand_rot,
            sampled_points,
            sampled_normals,
            goal_pos=goal_pos,
            goal_quat=goal_rot
        )
        # clip sensors
        sensors = torch.clamp(sensors, -clip_range, clip_range)
        sensors2g = torch.clamp(sensors2g, -clip_range, clip_range)
        dexrep_feat = torch.cat([sensors, sensors2g], dim=-1)
        return dexrep_feat

class SharedSurfH2GSensor(SharedSurf2GSensor):
    def __init__(self, args):
        import dexrep
        self.Sensor = dexrep.DexRep(sensor_type="surfh2g")
        self.task_name = args["task_name"]

        self.scaled_sampled_points = {}
        self.scaled_sampled_normals = {}

        self.obs_batch_obj_points = []
        self.obs_batch_obj_normals = []

        if "dexrep" in args.keys():
            self.sample_method = args["dexrep"]["sample_method"]
            self.sample_num_points = args["dexrep"]["sample_num_points"]
            self.BatchNormPnFeat = args["dexrep"]["batch_norm_pnfeat"]
        else:
            raise KeyError(f"sample_method not impletement")


class SharedSurfVV2GSensor(SharedSurf2GSensor):
    def __init__(self, args):
        import dexrep
        self.Sensor = dexrep.DexRep(sensor_type="surfvv2g")
        self.task_name = args["task_name"]

        self.scaled_sampled_points = {}
        self.scaled_sampled_normals = {}

        self.obs_batch_obj_points = []
        self.obs_batch_obj_normals = []

        if "dexrep" in args.keys():
            self.sample_method = args["dexrep"]["sample_method"]
            self.sample_num_points = args["dexrep"]["sample_num_points"]
            self.BatchNormPnFeat = args["dexrep"]["batch_norm_pnfeat"]
        else:
            raise KeyError(f"sample_method not impletement")

    def get_batch_perception_data(self,
                                  body_xpose,
                                  body_xquat,
                                  joints_site,
                                  hand_pos,
                                  hand_rot,
                                  sampled_points,
                                  sampled_normals,
                                  clip_range,
                                  goal_pos,
                                  goal_rot,
                                  mesh_names=None) -> torch.Tensor:
        sensors, sensors2g = self.Sensor.get_batch_perception_data(
            body_xpose,
            body_xquat,
            joints_site,
            hand_pos,
            hand_rot,
            sampled_points,
            sampled_normals,
            goal_pos=goal_pos,
            goal_quat=goal_rot
        )
        # clip sensors
        sensors = torch.clamp(sensors, -clip_range, clip_range)
        sensors2g = torch.clamp(sensors2g, -clip_range, clip_range)
        dexrep_feat = torch.cat([sensors, sensors2g], dim=-1)
        return dexrep_feat

class SharedDexRepDoubleSensor(SharedDexrep2GSensor):
    def __init__(self, args):
        super(SharedDexRepDoubleSensor, self).__init__(args)
        self.Sensor.sensot_type = "dexrep_double"

    def get_batch_perception_data(self,
                                  body_xpose,
                                  body_xquat,
                                  joints_site,
                                  hand_pos,
                                  hand_rot,
                                  sampled_points,
                                  sampled_normals,
                                  clip_range,
                                  goal_pos,
                                  goal_rot,
                                  mesh_names=None) ->torch.Tensor:
        sensors, pn_feat, sensors_goal, pn_feat_goal = self.Sensor.get_batch_perception_data(
            body_xpose,
            body_xquat,
            joints_site,
            hand_pos,
            hand_rot,
            sampled_points,
            sampled_normals,
            goal_pos=goal_pos,
            goal_quat=goal_rot
        )
        # clip sensors
        sensors = torch.clamp(sensors, -clip_range, clip_range)
        sensors_goal = torch.clamp(sensors_goal, -clip_range, clip_range)
        dexrep_feat = torch.cat([sensors, pn_feat,
                                 sensors_goal, pn_feat_goal], dim=-1)
        return dexrep_feat

class SharedDexrepV2GSensor(SharedDexrep2GSensor):
    def __init__(self, args):
        super(SharedDexrepV2GSensor, self).__init__(args)
        self.Sensor.sensot_type = "dexrep_VtoGoal"

class SharedDexrepVV2GSensor(SharedDexrep2GSensor):
    def __init__(self, args):
        super(SharedDexrepVV2GSensor, self).__init__(args)
        self.Sensor.sensot_type = "dexrep_VVtoGoal"


class SharedDexrepHand2GSensor(SharedDexrep2GSensor):
    def __init__(self, args):
        super(SharedDexrepHand2GSensor, self).__init__(args)
        self.Sensor.sensot_type = "dexrep_HtoGoal"


if __name__ == "__main__":
    obj_name = "ddg-ycb_002_master_chef_can"
    obj_path = f"/media/szn/PSSD/szn/My/Project/issac_manipulation/asset/reorient_up/{obj_name}/coacd/decomposed.obj"
    obj_mesh = trimesh.load(obj_path)
    # sample points
    object_verts = np.array(obj_mesh.vertices)
    object_normals = np.array(obj_mesh.vertex_normals)
    object_faces = np.array(obj_mesh.faces)
    fid, bc = pcu.sample_mesh_random(object_verts, object_faces, 2048)
    sampled_points = pcu.interpolate_barycentric_coords(object_faces, fid, bc, object_verts)
    sampled_normals = pcu.interpolate_barycentric_coords(object_faces, fid, bc, object_normals)
    # create ply
    obj_pcd = points2pcd(sampled_points, normals=sampled_normals)
    # visualize
    o3d.visualization.draw(obj_pcd)