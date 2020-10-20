import json
import numpy as np
import numpy.linalg as la
from os.path import join
import glob
import open3d as o3

id = 11


def load_data(path):
    global file_names_lidar, file_names_3dbox, pointcloud, points
    # Lidar Data
    root_path = path
    file_names_lidar = sorted(glob.glob(join(root_path, 'Lidar/*.npz')))
    # print(file_names_lidar)
    # 3D Bounding Boxes
    file_names_3dbox = sorted(glob.glob(join(root_path, '3DLabel/*.json')))
    # print(file_names_3dbox)
    # Images
    file_names_rgb = sorted(glob.glob(join(root_path, 'RGB/*.png')))
    # print(file_names_rgb[11])
    # Labels
    file_names_2dlabel = sorted(glob.glob(join(root_path, 'Label/*.png')))
    # print(file_names_2dlabel)
    pointcloud = np.load(file_names_lidar[id])
    pc_keys = list(pointcloud.keys())
    points = pointcloud['points']
    pc_o3 = o3.geometry.PointCloud()
    pc_o3.points = o3.utility.Vector3dVector(points)


load_data('./A2D2-Dataset/Dataset/')
EPSILON = 1.0e-10  # norm should not be small

with open('./A2D2-Dataset/cams_lidars.json', 'r') as f:
    config = json.load(f)


def get_axes_of_a_view(view):
    x_axis = view['x-axis']
    y_axis = view['y-axis']

    x_axis_norm = la.norm(x_axis)
    y_axis_norm = la.norm(y_axis)

    if (x_axis_norm < EPSILON or y_axis_norm < EPSILON):
        raise ValueError("Norm of input vector(s) too small.")

    # normalize the axes
    x_axis = x_axis / x_axis_norm
    y_axis = y_axis / y_axis_norm

    # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)

    # create orthogonal z-axis
    z_axis = np.cross(x_axis, y_axis)

    # calculate and check y-axis and z-axis norms
    y_axis_norm = la.norm(y_axis)
    z_axis_norm = la.norm(z_axis)

    if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
        raise ValueError("Norm of view axis vector(s) too small.")

    # make x/y/z-axes orthonormal
    y_axis = y_axis / y_axis_norm
    z_axis = z_axis / z_axis_norm

    return x_axis, y_axis, z_axis


def get_origin_of_a_view(view):
    return view['origin']


def get_transform_to_global(view):
    # get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)

    # get origin
    origin = get_origin_of_a_view(view)
    transform_to_global = np.eye(4)

    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis

    # origin
    transform_to_global[0:3, 3] = origin

    return transform_to_global


def get_transform_from_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    trans = np.eye(4)
    rot = np.transpose(transform_to_global[0:3, 0:3])
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])

    return trans


def transform_from_to(src, target):
    transform = np.dot(get_transform_from_global(target),
                       get_transform_to_global(src))

    return transform


id_2 = 781

# real image coordinates
r = pointcloud['row'][id_2]
c = pointcloud['col'][id_2]

# point
p = pointcloud['points'][id_2]
# print(p)
# p = np.append(p, 1)
# view_cam = config['cameras']['front_center']['view']
# view_lidar = config['cameras']['front_center']['view']
# transform_matr = transform_from_to(view_lidar, view_cam)
# # transform_matr = get_transform_to_global(view_lidar)
# transformed_pt = np.dot(p, transform_matr.T)
# # transformed = np.array([transformed[1], -transformed[2], transformed[0]])
# transformed_pt = p
transformed_pt = np.array([p[1], p[2], p[0]])
# print(p)
print(transformed_pt)
# print(p)

cam = np.array([
    [-1687.3369140625,
     0.0,
     965.43414055823814],
    [0.0,
     -1783.428466796875,
     684.4193604186803],
    [0.0,
     0.0,
     1.0]])

cam_2 = np.array([
    [
        1844.1774422790927,
        0.0,
        964.42990523863864
    ],
    [
        0.0,
        1841.5212239377258,
        679.5331911948183

    ],
    [
        0.0,
        0.0,
        1.0
    ]
])

res = np.dot(cam, transformed_pt)
# cam = cam.transpose()


pix_y = (transformed_pt[0] * cam[0][0])/transformed_pt[2] + cam[0][2]
pix_x = (transformed_pt[1] * cam[1][1])/transformed_pt[2] + cam[1][2]

res = res/res[2]

print("Self Multiplication: ", [res[1], res[0]])
print("OpenCV: ", [pix_x, pix_y])
print("Actual results: ", [r, c])
print("lidar_id: ", pointcloud['lidar_id'][id_2])
