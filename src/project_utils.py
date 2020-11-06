import json
import numpy as np
import open3d as o3
from os.path import join
import glob
import cv2

config_path = './A2D2-Dataset/cams_lidars.json'


def load_config(path):
    # Config File
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def skew_sym_matrix(u):
    return np.array([[0, -u[2], u[1]],
                     [u[2], 0, -u[0]],
                     [-u[1], u[0], 0]])


def axis_angle_to_rotation_mat(axis, angle):
    return np.cos(angle) * np.eye(3) + \
           np.sin(angle) * skew_sym_matrix(axis) + \
           (1 - np.cos(angle)) * np.outer(axis, axis)


# reads bounding boxes from file_name_bboxes
def read_bounding_boxes(file_name_bboxes):
    # open the file
    with open(file_name_bboxes, 'r') as f:
        bboxes = json.load(f)

    boxes = []  # a list for containing bounding boxes
    # print(bboxes.keys())

    for bbox in bboxes.keys():
        bbox_read = {}  # a dictionary for a given bounding box
        bbox_read['class'] = bboxes[bbox]['class']
        bbox_read['truncation'] = bboxes[bbox]['truncation']
        bbox_read['occlusion'] = bboxes[bbox]['occlusion']
        bbox_read['alpha'] = bboxes[bbox]['alpha']
        bbox_read['top'] = bboxes[bbox]['2d_bbox'][0]
        bbox_read['left'] = bboxes[bbox]['2d_bbox'][1]
        bbox_read['bottom'] = bboxes[bbox]['2d_bbox'][2]
        bbox_read['right'] = bboxes[bbox]['2d_bbox'][3]
        bbox_read['center'] = np.array(bboxes[bbox]['center'])
        bbox_read['size'] = np.array(bboxes[bbox]['size'])
        angle = bboxes[bbox]['rot_angle']
        axis = np.array(bboxes[bbox]['axis'])
        bbox_read['rotation'] = axis_angle_to_rotation_mat(axis, angle)
        bbox_read['axis'] = bboxes[bbox]['axis']
        bbox_read['angle'] = angle
        boxes.append(bbox_read)

    return boxes


# Extracts the points of a bbox
def get_points(bbox):
    half_size = bbox['size'] / 2.

    if half_size[0] > 0:
        # calculate unrotated corner point offsets relative to center
        brl = np.asarray([-half_size[0], +half_size[1], -half_size[2]])
        bfl = np.asarray([+half_size[0], +half_size[1], -half_size[2]])
        bfr = np.asarray([+half_size[0], -half_size[1], -half_size[2]])
        brr = np.asarray([-half_size[0], -half_size[1], -half_size[2]])
        trl = np.asarray([-half_size[0], +half_size[1], +half_size[2]])
        tfl = np.asarray([+half_size[0], +half_size[1], +half_size[2]])
        tfr = np.asarray([+half_size[0], -half_size[1], +half_size[2]])
        trr = np.asarray([-half_size[0], -half_size[1], +half_size[2]])

        # rotate points
        points = np.asarray([brl, bfl, bfr, brr, trl, tfl, tfr, trr])
        points = np.dot(points, bbox['rotation'].T)

        # add center position
        points = points + bbox['center']

    return points


# changes the rotation matrix of a 3d bbox according to a rotation angle
def update_rotation(box, angle):
    axis = box['axis']
    if axis[2] > 0:
        angle = box['angle'] + angle
    else:
        angle = box['angle'] - angle
    box["rotation"] = axis_angle_to_rotation_mat(axis, angle)
    return box


# Extracts lines out 3d-bboxes
def get_bboxes_wire_frames(bboxes, linesets=None, color=None):
    num_boxes = len(bboxes)

    # initialize linesets, if not given
    if linesets is None:
        linesets = [o3.geometry.LineSet() for _ in range(num_boxes)]

    # set default color
    if color is None:
        # color = [1, 0, 0]
        color = [0, 0, 1]

    assert len(linesets) == num_boxes, "Number of linesets must equal number of bounding boxes"

    # point indices defining bounding box edges
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [0, 4], [1, 5], [2, 6], [3, 7],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [5, 2], [1, 6]]

    # loop over all bounding boxes
    for i in range(num_boxes):
        # get bounding box corner points
        points = get_points(bboxes[i])
        # update corresponding Open3d line set
        colors = [color for _ in range(len(lines))]
        line_set = linesets[i]
        line_set.points = o3.utility.Vector3dVector(points)
        line_set.lines = o3.utility.Vector2iVector(lines)
        line_set.colors = o3.utility.Vector3dVector(colors)

    return linesets


def undistort_image(image, cam_name):
    if cam_name in ['front_left', 'front_center', 'front_right', 'side_left', 'side_right', 'rear_center']:
        # get parameters from config file
        intr_mat_undist = np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = \
            np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = \
            np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']

        if lens == 'Fisheye':
            return cv2.fisheye.undistortImage(image, intr_mat_dist,
                                              D=dist_parms, Knew=intr_mat_undist)
        elif lens == 'Telecam':
            return cv2.undistort(image, intr_mat_dist,
                                 distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image


# adds bboxes to the PC-Object of Open3D library
def add_boxes_to_pointcloud(show=False):
    global box, bboxes, pointcloud, points
    pointcloud = np.load(file_names_lidar[id])
    pc_keys = list(pointcloud.keys())
    points = pointcloud['points']
    pc_o3 = o3.geometry.PointCloud()
    pc_o3.points = o3.utility.Vector3dVector(points)

    bboxes = read_bounding_boxes(file_names_3dbox[id])
    box = bboxes[box_id]
    entities_to_draw = [pc_o3]
    for bbox in bboxes:
        linesets = get_bboxes_wire_frames([bbox], color=(255, 0, 0))
        entities_to_draw.append(linesets[0])

    print(entities_to_draw)
    if show:
        o3.visualization.draw_geometries(entities_to_draw)


# enclose points in a predefined box
def find_points_in_box():
    global s, car
    ps = get_points(box)
    l = np.min(ps, axis=0)
    u = np.max(ps, axis=0)
    s = np.logical_and(points > l, points < u)
    s = np.all(s, axis=1)
    car = points[s]
    obj = car.copy()
    np.save('14_1', obj)


# PC with Object cut out. (See what object you work with)
def create_point_cloud_without_object(show=False, shift=30, axis=1):
    global car, box
    s_2 = np.array([not i for i in s])
    pc_without_car = points[s_2]
    car[:, axis] = car[:, axis] + shift
    box['center'][axis] = box['center'][axis] + shift
    # car = np.append(car, pc_without_car, axis=0)
    pcd_car = o3.geometry.PointCloud()
    pcd_car.points = o3.utility.Vector3dVector(car)
    pcd_without_car = o3.geometry.PointCloud()
    pcd_without_car.points = o3.utility.Vector3dVector(pc_without_car)
    linesets = get_bboxes_wire_frames([box], color=(255, 0, 0))

    entities = []
    entities.append(pcd_without_car)
    entities.append(pcd_car)
    entities.append(linesets[0])

    if show:
        o3.visualization.draw_geometries(entities)


# Pointcloud augmented
def augment_point_cloud(id):
    pointcloud_2 = np.load(file_names_lidar[id])
    points_2 = pointcloud_2['points']
    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(points_2)
    o3.visualization.draw_geometries([pcd])


config = load_config(config_path)

# Class colors in HEX
class_list = {
    "#ff0000": "Car 1",
    "#c80000": "Car 2",
    "#960000": "Car 3",
    "#800000": "Car 4",
    "#b65906": "Bicycle 1",
    "#963204": "Bicycle 2",
    "#5a1e01": "Bicycle 3",
    "#5a1e1e": "Bicycle 4",
    "#cc99ff": "Pedestrian 1",
    "#bd499b": "Pedestrian 2",
    "#ef59bf": "Pedestrian 3",
    "#ff8000": "Truck 1",
    "#c88000": "Truck 2",
    "#968000": "Truck 3",
    "#00ff00": "Small vehicles 1",
    "#00c800": "Small vehicles 2",
    "#009600": "Small vehicles 3",
    "#0080ff": "Traffic signal 1",
    "#1e1c9e": "Traffic signal 2",
    "#3c1c64": "Traffic signal 3",
    "#00ffff": "Traffic sign 1",
    "#1edcdc": "Traffic sign 2",
    "#3c9dc7": "Traffic sign 3",
    "#ffff00": "Utility vehicle 1",
    "#ffffc8": "Utility vehicle 2",
    "#e96400": "Sidebars",
    "#6e6e00": "Speed bumper",
    "#808000": "Curbstone",
    "#ffc125": "Solid line",
    "#400040": "Irrelevant signs",
    "#b97a57": "Road blocks",
    "#000064": "Tractor",
    "#8b636c": "Non-drivable street",
    "#d23273": "Zebra crossing",
    "#ff0080": "Obstacles / trash",
    "#fff68f": "Poles",
    "#960096": "RD restricted area",
    "#ccff99": "Animals",
    "#eea2ad": "Grid structure",
    "#212cb1": "Signal corpus",
    "#b432b4": "Drivable cobblestone",
    "#ff46b9": "Electronic traffic",
    "#eee9bf": "Slow drive area",
    "#93fdc2": "Nature object",
    "#9696c8": "Parking area",
    "#b496c8": "Sidewalk",
    "#48d1cc": "Ego car",
    "#c87dd2": "Painted driv. instr.",
    "#9f79ee": "Traffic guide obj.",
    "#8000ff": "Dashed line",
    "#ff00ff": "RD normal street",
    "#87ceff": "Sky",
    "#f1e6ff": "Buildings",
    "#60458f": "Blurred area",
    "#352e52": "Rain dirt"
}

# Modified camera matrices
cam_matrix_undist = np.array([
    [-1687.3369140625,
     0.0,
     965.43414055823814],
    [0.0,
     -1783.428466796875,
     684.4193604186803],
    [0.0,
     0.0,
     1.0]])

cam_matrix_dist = np.array([
    [-1844.1774422790927,
     0.0,
     964.42990523863864],
    [0.0,
     -1841.5212239377258,
     679.5331911948183],
    [0.0,
     0.0,
     1.0]])
