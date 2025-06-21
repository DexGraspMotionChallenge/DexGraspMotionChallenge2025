
import numpy as np
import torch
import math

from pytorch3d.transforms import euler_angles_to_matrix,matrix_to_euler_angles
from scipy.spatial.transform import Rotation as R

def modify_hand_trajectory(seq_grasps, gamma=0.1):
    """
    修改 shadow hand 轨迹：
    1. 保持手腕（前6维）不变。
    2. 手指部分（后22维）采用分段插值：
       - 前 T-5 帧：从 flat 状态线性缓慢增加至 gamma 程度，
       - 最后 5 帧：从 gamma 快速过渡到最终的抓紧状态。

    参数：
      seq_grasps: numpy 数组，形状为 (N, T, 28)
      gamma: 前 T-5 帧结束时的插值因子，控制变化的缓慢程度（默认0.1）

    返回：
      new_seq: 修改后的轨迹数组，形状同样为 (N, T, 28)
    """
    new_seq = np.copy(seq_grasps)
    N, T, D = new_seq.shape
    if D != 28:
        raise ValueError("数据维度必须为28！")

    # 初始flat手状态（通常第0帧即为flat状态）
    flat_finger = seq_grasps[:, 0, 6:28]    # (N, 22)
    # 目标抓紧状态使用最后一帧的手指参数
    final_finger = seq_grasps[:, -1, 6:28]    # (N, 22)

    # 构造每一帧的插值因子alpha，其维度为 (T,)
    alpha = np.zeros(T)

    # 前部分：帧 0 到 T-5，缓慢变化从 0 到 gamma
    # 注意：如果T-5不足2帧，则保持默认
    if T- 5 > 1:
        for t in range(T - 5):
            alpha[t] = (t / (T - 5 - 1)) * gamma
    else:
        alpha[:T - 5] = 0

    # 后部分：最后5帧，alpha从 gamma 快速增长至 1
    num_last = 5
    if num_last > 1:
        for i, t in enumerate(range(T - 5, T)):
            alpha[t] = gamma + (i / (num_last - 1)) * (1 - gamma)
    else:
        alpha[T - 5:] = 1

    # 对每个帧，按照alpha值插值：new = (1 - alpha) * flat + alpha * final
    # 利用广播，axis 对应帧维度
    for t in range(T):
        new_seq[:, t, 6:28] = (1 - alpha[t]) * flat_finger + alpha[t] * final_finger

    # 手腕部分（前6维）保持不变，new_seq[:, :, 0:6] 没有修改
    return new_seq


def downsampling_trajectory(seq_grasps, end_idx=20):
    """
    seq_grasps (B,T,28)
    """
    # 要删除的帧索引（从前20帧中每隔1帧）
    remove_indices = np.arange(0, end_idx, 2)  # array([ 0,  2,  4, ..., 18])

    # 删除这些帧，axis=1 表示在帧维度上删除
    sample_traj = np.delete(seq_grasps, remove_indices, axis=1)
    return sample_traj



def compute_h2o_minimum_vec(hand_points, object_points):
    """
    计算手部点云和物体点云之间的全局最小距离
    hand_points: (T, N, 3) PyTorch Tensor
    object_points: (1, M, 3) PyTorch Tensor
    """
    T = len(hand_points)
    if isinstance(hand_points,np.ndarray):
        hand_points = torch.tensor(hand_points,dtype=torch.float32)
    if isinstance(hand_points,np.ndarray):
        object_points = torch.tensor(object_points,dtype=torch.float32)

    dist_matrix = torch.cdist(hand_points, object_points)  # 计算所有点对的欧式距离 (N, M)
    h2o_dist,index_of_obj = torch.min(dist_matrix,dim=2)  # 找到全局最小距离
    indices = index_of_obj.unsqueeze(-1).expand(-1, -1, 3)  # (T, 2000, 3)，为 gather 做准备
    if object_points.shape[0]==1:
        obj_corresponding_points = torch.gather(object_points.expand(T,-1,-1), dim=1, index=indices)
    else:
        obj_corresponding_points = torch.gather(object_points, dim=1, index=indices)
    h2o_vec = obj_corresponding_points-hand_points
    return h2o_vec, h2o_dist



def augment_grasp_data(data_dict, target_N=50):
    N = data_dict['grasp_seqs'].shape[0]
    if N >= target_N:
        return data_dict

    num_to_augment = target_N - N
    sample_indices = np.random.randint(0, N, size=num_to_augment)

    # 复制所有 key，其 shape[0] == N 的字段
    augmented_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.shape[0] == N:
            augmented_data = value[sample_indices].copy()
            augmented_dict[key] = np.concatenate([value, augmented_data], axis=0)
        else:
            augmented_dict[key] = value  # 不扩展

    # 特别处理 grasp_seqs 第一帧的增强
    grasp_seqs = augmented_dict['grasp_seqs']
    # N_new = grasp_seqs.shape[0]

    # 找出刚刚增强的那部分轨迹（后 num_to_augment 个）
    augmented_grasps = grasp_seqs[-num_to_augment:]  # (num_to_augment, T, 28)

    # 扰动（translation YZ + 欧拉角）
    delta_translation_y = np.random.uniform(-0.02, 0.03, size=num_to_augment)
    delta_translation_z = np.random.uniform(-0.02, 0.02, size=num_to_augment)
    # delta_euler_xyz = np.random.uniform(-0.2, 0.2, size=(num_to_augment, 3))

    augmented_grasps[:, 0, 1] += delta_translation_y
    augmented_grasps[:, 0, 2] += delta_translation_z
    # augmented_grasps[:, 0, 3:6] += delta_euler_xyz


    # 更新回 dict 中
    augmented_dict['grasp_seqs'][-num_to_augment:] = augmented_grasps

    return augmented_dict


# 获取手掌方向向量
def get_palm_dirs(euler_batch):  # (B, 3)
    """
    euler_batch:(B, 3)
    """
    # R = euler_xyz_to_matrix(euler_batch)  # (B, 3, 3)
    R =  euler_angles_to_matrix(euler_batch, convention="XYZ")#(B,3,3)

    v_palm = torch.tensor([0., -1., 0.], device=euler_batch.device).view(1, 3, 1).expand(euler_batch.shape[0], 3, 1)  # (B, 3, 1)
    palm_dirs = torch.bmm(R, v_palm).squeeze(-1)  # (B, 3)
    return palm_dirs



# 计算手掌方向与六个主轴方向的夹角（取最大余弦相似度）
def get_nearest_axis(palm_dirs):
    # (6, 3) for [±X, ±Y, ±Z]
    axis_vecs = torch.tensor([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ], dtype=torch.float, device=palm_dirs.device)  # (6, 3)

    # palm_dirs: (B, 3) → (B, 1, 3), axis_vecs: (1, 6, 3)
    palm_norm = palm_dirs / palm_dirs.norm(dim=-1, keepdim=True)
    axis_norm = axis_vecs / axis_vecs.norm(dim=-1, keepdim=True)

    dots = torch.matmul(palm_norm.unsqueeze(1), axis_norm.T.unsqueeze(0))  # (B, 6)
    best_match = dots.argmax(dim=-1)  # (B,)
    return best_match  # 返回方向索引：0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z


def get_alignment_rotation_matrix(direction_index):  # (B,)
    B = direction_index.shape[0]
    R_out = torch.eye(3, device=direction_index.device).repeat(B, 1, 1)  # default: identity

    # 旋转矩阵集合
    def rot(axis, angle_deg):
        angle = math.radians(angle_deg)
        if axis == 'x':
            return torch.tensor([
                [1, 0, 0],
                [0, math.cos(angle), -math.sin(angle)],
                [0, math.sin(angle), math.cos(angle)]
            ], dtype=torch.float, device=direction_index.device)
        elif axis == 'y':
            return torch.tensor([
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)]
            ], dtype=torch.float, device=direction_index.device)
        elif axis == 'z':
            return torch.tensor([
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1]
            ], dtype=torch.float, device=direction_index.device)

    for i in range(B):
        d = direction_index[i].item()
        if d == 0:  # +X → rot Y 90°
            R_out[i] = rot('y', 90)
        elif d == 1:  # -X → rot Y -90°
            R_out[i] = rot('y', -90)
        elif d == 2:  # +Y → rot X -90°
            R_out[i] = rot('x', -90)
        elif d == 3:  # -Y → rot X 90°
            R_out[i] = rot('x', 90)
        elif d == 4:  # +Z → rot X 180°
            R_out[i] = rot('x', 180)
        elif d == 5:  # -Z → 无需旋转
            pass
    return R_out  # (B, 3, 3)


# 对 Trajectory旋转，并返回对应的旋转矩阵
def rotate_trajs_and_object_to_zneg_vectorized(trajs, object_points=None):
    """
    trajs：（B,T,28）
    """
    B, T, _ = trajs.shape
    # device = trajs.device

    # Step 1: 计算每条轨迹第一帧手掌方向
    start_euler = trajs[:, 0, 3:6]  # (B, 3)
    palm_dirs = get_palm_dirs(start_euler)  # (B, 3)

    # Step 2: 找到最接近的主方向，并构造对应旋转矩阵
    nearest_dir = get_nearest_axis(palm_dirs)  # (B,)
    R_align = get_alignment_rotation_matrix(nearest_dir)  # (B, 3, 3)

    # Step 3: 平移旋转，shape: (B, T, 3)
    trans = trajs[:, :, 0:3]  # (B, T, 3)
    trans_rot = torch.matmul(R_align.unsqueeze(1), trans.unsqueeze(-1)).squeeze(-1)  # (B, T, 3)

    # Step 4: 姿态旋转，先转换为旋转矩阵
    euler_all = trajs[:, :, 3:6].reshape(B * T, 3)
    # R_ori = euler_xyz_to_matrix(euler_all).reshape(B, T, 3, 3)  # (B, T, 3, 3)
    R_ori =euler_angles_to_matrix(euler_all, convention="XYZ").reshape(B, T, 3, 3)#(B, T, 3,3)

    R_align_expand = R_align.unsqueeze(1)  # (B, 1, 3, 3)
    R_new = torch.matmul(R_align_expand, R_ori)  # (B, T, 3, 3)

    # 再转回欧拉角
    R_new_flat = R_new.reshape(B * T, 3, 3)
    # euler_new = matrix_to_euler_xyz(R_new_flat).reshape(B, T, 3)  # (B, T, 3)
    euler_new = matrix_to_euler_angles(R_new_flat,convention="XYZ").reshape(B, T, 3)  # (B, T, 3)

    # Step 5: 替换原始数据
    new_trajs = trajs.clone()
    new_trajs[:, :, 0:3] = trans_rot
    new_trajs[:, :, 3:6] = euler_new

    # Step 6: 物体旋转，(B, N, 3)
    if object_points is not None:
        obj_rot = torch.matmul(R_align, object_points.transpose(1, 2)).transpose(1, 2)
        return new_trajs, obj_rot

    return new_trajs, R_align


def unwrap_euler_batch_torch(batch_traj, angle_slice=(3, 6)):
    """
    在 PyTorch 中向量化解包批量轨迹中的欧拉角部分，消除 ±π 跳变。

    参数：
        batch_traj: (N, T, 28) 的 torch.Tensor，单位为弧度
        angle_slice: 欧拉角维度范围，默认为 (3,6)

    返回：
        解包后的 torch.Tensor，shape 同输入
    """
    a, b = angle_slice
    traj = batch_traj.clone()
    euler = traj[:, :, a:b]  # (N, T, 3)

    # (N, T-1, 3) 的相邻帧角度差
    delta = euler[:, 1:, :] - euler[:, :-1, :]

    # 找出超过 pi 的差值，并计算需要加减 2pi 的次数
    correction = torch.round(delta / (2 * torch.pi))  # (N, T-1, 3)

    # 构建累加修正量：形状为 (N, T, 3)
    correction_full = torch.zeros_like(euler)
    correction_full[:, 1:, :] = torch.cumsum(correction, dim=1)

    # 应用解包修正
    unwrapped_euler = euler - correction_full * 2 * torch.pi
    traj[:, :, a:b] = unwrapped_euler
    return traj

def unwrap_euler_batch_vectorized(batch_traj, angle_slice=(3, 6)):
    """
    向量化方式批量对 (N, T, 28) 轨迹中欧拉角部分进行时间轴上的 unwrap。
    参数：
        batch_traj: shape (N, T, 28)
        angle_slice: 欧拉角维度范围 (默认为 3:6)

    返回：
        解包后的 batch_traj，shape 保持 (N, T, 28)
    """
    batch_traj = batch_traj.copy()
    a, b = angle_slice

    # 取出欧拉角部分，reshape 成 (N*3, T)
    euler_batch = batch_traj[:, :, a:b]              # (N, T, 3)
    euler_reshaped = np.transpose(euler_batch, (0, 2, 1))  # (N, 3, T)
    euler_flat = euler_reshaped.reshape(-1, euler_batch.shape[1])  # (N*3, T)

    # 沿时间轴解包
    unwrapped_flat = np.unwrap(euler_flat, axis=1)   # (N*3, T)

    # 恢复成原始形状
    unwrapped = unwrapped_flat.reshape(euler_reshaped.shape)  # (N, 3, T)
    unwrapped = np.transpose(unwrapped, (0, 2, 1))  # (N, T, 3)

    # 替换到原始轨迹
    batch_traj[:, :, a:b] = unwrapped

    return batch_traj


def select_diverse_trajectories(trajs, num_select=30):
    """
    从 (N, T, 28) 的轨迹中，选出在第一帧手腕姿态差异性最大的 num_select 条轨迹。
    """
    N, T, D = trajs.shape
    assert D >= 6, "需要包含位置(3)和欧拉角(3)"

    # 提取第 0 帧的 translation 和 rotation（欧拉角）
    poses = trajs[:, 0, :6]  # shape (N, 6)

    # 分别处理 translation 和 rotation
    translations = poses[:, :3]  # (N, 3)
    eulers = poses[:, 3:6]       # (N, 3)

    # 将欧拉角转为旋转矩阵 or 四元数（避免周期问题）
    rots = R.from_euler("XYZ", eulers).as_quat()  # (N, 4)

    # 构建特征向量（可拼接 translation 和 quat）
    features = np.concatenate([translations, rots], axis=1)  # (N, 7)

    # 计算 pairwise L2 distance
    dist_matrix = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)  # (N, N)

    # 用最大最小距离贪心法选出 num_select 个最分散的轨迹
    selected = [np.random.randint(0, N)]  # 从随机一个点开始
    for _ in range(1, num_select):
        remaining = list(set(range(N)) - set(selected))
        min_dists = np.min(dist_matrix[remaining][:, selected], axis=1)
        next_idx = remaining[np.argmax(min_dists)]
        selected.append(next_idx)

    return selected  # 返回选中的 N 个轨迹索