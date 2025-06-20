from open3d import visualization as o3dv
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from open3d import utility as o3du
from open3d import geometry as o3dg
import torch
import cv2 as cv
import trimesh
import scenepic as sp
from open3d import geometry as o3dg


def crop(image):
    images_c = image.mean(axis=2)
    x_index = np.nonzero(1-images_c.mean(axis=1))[0]
    x_window = [x_index.min(), x_index.max()+2]

    y_index = np.nonzero(1-images_c.mean(axis=0))[0]
    y_window = [y_index.min(),y_index.max()+2]

    return image[x_window[0]:x_window[1], y_window[0]:y_window[1],:]



def custom_vis_screen_capcture(mesh_list, key_str, save_path, view_k=2):
    """
    :param mesh_list: the lists contains N meshes
    :param key_str: the name for the save images
    :param save_path: image save path
    :param view_k: num of view needed to be visualized.
    :return:
    """

    vis = o3dv.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    for m_i in mesh_list:
        vis.add_geometry(m_i)

    ctr.rotate(0.0, -180.0) #-500.
    if len(mesh_list)>1:
        ctr.scale(-12.0)
    else:
        ctr.scale(5.0)

    view_rots = [200, 300.0, 600.0] #[540.0, 240.0, 540.0]
    for i in range(view_k):
        rots= view_rots[i]
        ctr.rotate(rots, 0.0)
        image = vis.capture_screen_float_buffer(True)
        image = crop(np.asarray(image))

        plt.imshow(image)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, key_str + '_rot{}.png'.format(i)))


        # plt.show()

    vis.destroy_window()

    a = 1
def Gray2Jet(color_map):
    gray_value = color_map.cpu().numpy() * 255  # (2048)
    color_values = cv.applyColorMap(np.uint8(gray_value), cv.COLORMAP_JET).reshape(color_map.size(0), 3)
    return color_values / 255

def contact_to_color(contact_map, color='red'):
    assert len(contact_map.size()) == 1
    contact_mask = (contact_map >= 0.1)

    # contact_color_map = torch.ones(contact_map.size(0), 3)

    contact_color_map = torch.ones(contact_map.size(0), 3) * torch.tensor([0., 1., 1.])
    contact_color_map = contact_color_map.cuda()
    if color == 'red':
        contact_color_map[contact_mask, 2] = 1. - contact_map[contact_mask]
        contact_color_map[contact_mask, 1] = 1. - contact_map[contact_mask]
        contact_color_map[contact_mask, 0] = 0. + contact_map[contact_mask]

    if color == 'green':
        contact_color_map[contact_mask, 2] = 1. - contact_map[contact_mask]

    if color == 'blue':
        contact_color_map[contact_mask, 1] = 1. - contact_map[contact_mask]

    if color == 'jet':
        contact_color_map = torch.zeros(contact_map.size(0), 3)
        jet_color_map = torch.from_numpy(Gray2Jet(contact_map))
        contact_color_map[:, 0] = jet_color_map[:, 2]
        contact_color_map[:, 1] = jet_color_map[:, 1]
        contact_color_map[:, 2] = jet_color_map[:, 0]
    a = 1

    return contact_color_map.float()

def obj_mesh_instance(data, contact=None, trans=0., colors ='red'):
    obj_mesh = o3dg.TriangleMesh()
    obj_verts = data['obj_verts_gt'].cpu().squeeze().numpy()
    obj_verts[:,1]+=trans

    obj_mesh.vertices = o3du.Vector3dVector(obj_verts)
    obj_mesh.triangles = o3du.Vector3iVector(data['obj_faces'].cpu().squeeze().numpy())

    if torch.is_tensor(contact):
        color_map = contact_to_color(contact.squeeze(),color=colors)
        obj_mesh.vertex_colors = o3du.Vector3dVector(color_map.cpu().squeeze().numpy())
    else:
        obj_mesh.paint_uniform_color([0., 1., 1.])

    obj_mesh.compute_vertex_normals()
    return obj_mesh



def hand_mesh_instance(verts, faces, trans=0., uni_color=[0.8, 0.1, 0]):
    hand_mesh = o3dg.TriangleMesh()
    verts[:, 1] += trans

    hand_mesh.vertices = o3du.Vector3dVector(verts.cpu().squeeze().detach().numpy()+trans)
    hand_mesh.triangles = o3du.Vector3iVector(faces)
    hand_mesh.compute_vertex_normals()
    hand_mesh.paint_uniform_color(uni_color)
    hand_mesh.compute_vertex_normals()
    return hand_mesh

def pcd_instance(pcd, trans=0., contact=None, color=[0.8, 0.1, 0], c_color='jet'):
    if torch.is_tensor(pcd):
        pcd = pcd.squeeze().detach().cpu().numpy()
    pcd[:, 2] += trans
    pointcloud = o3dg.PointCloud()
    pointcloud.points = o3du.Vector3dVector((pcd))
    if contact is not None:
        if torch.is_tensor(contact):
            contact = contact.detach().cpu().numpy()
        color_map = contact_to_color(contact.squeeze(), color=c_color)
        pointcloud.colors = o3du.Vector3dVector(color_map)
    else:
        pointcloud.paint_uniform_color(color)

    return pointcloud


def html_antmation_save(hand_mesh_list, obj_mesh_list,extra_mesh_list=None, name='test'):

    FOR1 = o3dg.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    FOR1 = trimesh.Trimesh(vertices=np.asarray(FOR1.vertices),faces=np.asarray(FOR1.triangles)
                           ,vertex_colors=np.asarray(FOR1.vertex_colors))

    grasp_anim = sp_animation()
    for i,meshes_i in enumerate(zip(hand_mesh_list,obj_mesh_list)):
        hand_mesh,obj_mesh = meshes_i

        if extra_mesh_list is not None:
            extra_mesh = extra_mesh_list[i]
            grasp_anim.add_frame([hand_mesh, obj_mesh, FOR1, extra_mesh], ['hand', 'obj', 'XYZaxis','extra'])
        else:
            grasp_anim.add_frame([hand_mesh, obj_mesh, FOR1], ['hand', 'obj', 'XYZaxis'])


    grasp_anim.save_animation(osp.join('./', "{}.html".format(name,i)))


class sp_animation():
    def __init__(self,
                 width = 1600,
                 height = 1600,
                 ):
        super(sp_animation, self).__init__()
        shading = sp.Shading(bg_color=np.array([1.0, 1.0, 1.0]))

        self.scene = sp.Scene()
        self.main = self.scene.create_canvas_3d(width=width, height=height,shading=shading)
        self.colors = sp.Colors

    def meshes_to_sp(self,meshes_list, layer_names):

        sp_meshes = []

        for i, m in enumerate(meshes_list):
            params = {'vertices' : np.asarray(m.vertices).astype(np.float32),
                      'normals' :np.asarray(m.vertex_normals).astype(np.float32), #m.estimate_vertex_normals().astype(np.float32),
                      'triangles' : np.asarray(m.faces).astype(np.float32),
                      'colors' : np.asarray(m.visual.vertex_colors).astype(np.float32)[:,:3]/254
                      }
            # params = {'vertices' : m.v.astype(np.float32), 'triangles' : m.f, 'colors' : m.vc.astype(np.float32)}
            # sp_m = sp.Mesh()
            sp_m = self.scene.create_mesh(layer_id = layer_names[i])
            sp_m.add_mesh_with_normals(**params)
            if layer_names[i] == 'ground_mesh':
                sp_m.double_sided=True
            sp_meshes.append(sp_m)

        return sp_meshes

    def add_frame(self,meshes_list_ps, layer_names):

        meshes_list = self.meshes_to_sp(meshes_list_ps, layer_names)
        if not hasattr(self,'focus_point'):
            # self.focus_point = meshes_list_ps[1].v.mean(0)
            self.focus_point = np.array([0.,0.,0.])

            # center = self.focus_point
            # center[2] = 4
            # rotation = sp.Transforms.rotation_about_z(0)
            # self.camera = sp.Camera(center=center, rotation=rotation, fov_y_degrees=30.0)

        main_frame = self.main.create_frame(focus_point=self.focus_point)
        for i, m in enumerate(meshes_list):
            # self.main.set_layer_settings({layer_names[i]:{}})
            main_frame.add_mesh(m)

    def save_animation(self, sp_anim_name):
        self.scene.link_canvas_events(self.main)
        self.scene.save_as_html(sp_anim_name, title=sp_anim_name.split('/')[-1])