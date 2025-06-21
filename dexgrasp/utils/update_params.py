import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, axis_angle_to_matrix


def get_transform_Rz(start_trans):
    phi = torch.atan2(start_trans[:, 0], start_trans[:, 1])  # (B,)

    # 构造批量的绕 Z 轴旋转矩阵
    # 利用 axis-angle 表示法：每个样本的旋转轴均为 Z 轴，只需将对应的角度赋值给 z 分量
    B = start_trans.shape[0]
    angle_axis = torch.zeros((B, 3), dtype=start_trans.dtype, device=start_trans.device)
    angle_axis[:, 2] = phi
    Rz = axis_angle_to_matrix(angle_axis)  # (1, 3, 3)

    return Rz

def update_grasp_wrist_params_batch(t, theta_euler, Rz):
    """
    更新批量抓取手势参数，使得手腕 translation 经绕 Z 轴旋转后落在 YZ 平面上。

    参数:
      t: (B, 3) tensor，每行为 wrist translation (x, y, z)
      theta_euler: (B, 3) tensor，每行为 wrist orientation (XYZ 欧拉角，单位为弧度)

    返回:
      t_new: (B, 3) tensor，更新后的 wrist translation
      theta_new: (B, 3) tensor，更新后的 wrist orientation (XYZ 欧拉角)
    """
    # 计算每个样本绕 Z 轴旋转的角度 phi，使得旋转后 x 分量为 0
    # 由条件 x*cos(phi) - y*sin(phi)=0 可得 phi = atan2(x, y)
    # phi = torch.atan2(t[:, 0], t[:, 1])  # (B,)
    #
    # # 构造批量的绕 Z 轴旋转矩阵
    # # 利用 axis-angle 表示法：每个样本的旋转轴均为 Z 轴，只需将对应的角度赋值给 z 分量
    # B = t.shape[0]
    # angle_axis = torch.zeros((B, 3), dtype=t.dtype, device=t.device)
    # angle_axis[:, 2] = phi
    # Rz = axis_angle_to_matrix(angle_axis)  # (B, 3, 3)

    # 更新 wrist translation：t_new = Rz * t
    t_new = torch.bmm(Rz, t.unsqueeze(-1)).squeeze(-1)

    # 将原始的 wrist orientation（欧拉角）转换为旋转矩阵（使用 XYZ 顺序）
    R_orig = euler_angles_to_matrix(theta_euler, convention="XYZ")  # (B, 3, 3)

    # 更新 wrist orientation：整个手势绕 Z 轴旋转，即 R_new = Rz * R_orig
    R_new = torch.bmm(Rz, R_orig)

    # 将更新后的旋转矩阵转换回 XYZ 欧拉角
    theta_new = matrix_to_euler_angles(R_new, convention="XYZ")

    return t_new, theta_new



def update_seq_grasp_direction_batch(params):
    """
    params: (N,T, 28)
    """
    N,T = params.size()[:2]
    start_trans = params[:,0,:3] #(N, 3)
    Rz = get_transform_Rz(start_trans) #(N, 3, 3)
    Rz_batch = Rz.unsqueeze(1).repeat(1,T,1,1).reshape(-1,3,3) #(N*T,3,3)

    params_trans = params[:,:,:3].reshape(-1,3) #(N*T,3)
    params_euler = params[:,:,3:6].reshape(-1,3) #(N*T,3)

    params_trans_new, params_euler_new = update_grasp_wrist_params_batch(params_trans,params_euler, Rz_batch) #(N*T,3), (N*T,3, 3)
    params[:, :, :6] = torch.cat([params_trans_new.reshape(N,T,3),params_euler_new.reshape(N,T,3)], dim=-1)
    params = params.reshape(N,T,-1)
    return params, Rz



# 示例测试
if __name__ == '__main__':
    B = 5
    # 示例数据：随机的 wrist translation
    t = torch.tensor([[0.1, 0.2, 0.3],
                      [0.2, 0.1, 0.4],
                      [-0.1, 0.3, 0.2],
                      [0.0, 0.5, 0.3],
                      [0.3, -0.2, 0.1]], dtype=torch.float32)
    # 初始 wrist orientation 均为 0，即手腕朝向与 Z 轴一致
    theta_euler = torch.zeros((B, 3), dtype=torch.float32)

    t_new, theta_new = update_grasp_params_batch(t, theta_euler)
    print("更新后的 wrist translation:\n", t_new)
    print("更新后的 wrist orientation (XYZ):\n", theta_new)
