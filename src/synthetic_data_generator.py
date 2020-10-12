from os.path import join
import glob
import numpy as np
import open3d as o3
import json
import matplotlib.pyplot as plt

id = 11
box_id = 5


# Loads data from A2D2. Assigns the points in the pointcloud to the variable "points".
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


# o3.visualization.draw_geometries([pc_o3])

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


# calculate the rotation angle of a new position
def new_position_rotation_angle(box, new_pos):
    global old_pos
    old_pos = box['center'][:2]
    print("old: ", old_pos)
    print("new: ", new_pos)
    angle_cos = np.dot(old_pos, new_pos) / \
                (np.linalg.norm(old_pos) * np.linalg.norm(new_pos))
    angle = np.arccos(angle_cos)
    print("angle: ", angle * 180 / np.pi)
    if old_pos[1] < new_pos[1]:
        return angle
    else:
        return -angle


# Extracts lines out 3d-bboxes
def _get_bboxes_wire_frames(bboxes, linesets=None, color=None):
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
        linesets = _get_bboxes_wire_frames([bbox], color=(255, 0, 0))
        entities_to_draw.append(linesets[0])

    # print(entities_to_draw)
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
    # np.save('14_1', obj)


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
    linesets = _get_bboxes_wire_frames([box], color=(255, 0, 0))

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


# propose Positions for the new point cloud
def generate_positions(points):
    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])
    rand_x = np.random.randint(x_min + 2, x_max)
    # rand_y = np.random.randint(y_min, y_max)
    rand_y = np.random.randint(old_pos[1], - old_pos[1])
    return rand_x, rand_y


# check proposed Positions
def check_positions(coords, box, points, ground_tolerance=0.11, size_tolerance=0.5):
    box_3d = box.copy()
    box_3d['center'] = box['center'].copy()
    angle = new_position_rotation_angle(box, coords)
    box_3d['center'][0] = coords[0]
    box_3d['center'][1] = coords[1]
    coords = np.asarray(coords)
    update_rotation(box_3d, angle)
    update_rotation(box, angle)

    ps = get_points(box_3d)
    lower_corner = np.min(ps, axis=0)
    upper_corner = np.max(ps, axis=0)

    # ground tolerance
    lower_corner[2] = lower_corner[2] + ground_tolerance

    # size tolerance
    lower_corner[0] = lower_corner[0] - size_tolerance
    lower_corner[1] = lower_corner[1] - size_tolerance
    upper_corner[0] = upper_corner[0] + size_tolerance
    upper_corner[1] = upper_corner[1] + size_tolerance

    condition = np.logical_and(points > lower_corner, points < upper_corner)
    condition = np.all(condition, axis=1)
    s = points[condition]
    if len(s) > 0:
        return False
    else:
        return True


# Find a good Position
def find_position(global_points, bbox_3d):
    bad_position = True
    while bad_position == True:
        new_position = generate_positions(global_points)
        bad_position = not check_positions(new_position, bbox_3d, global_points)

    return new_position


# plot pointcloud with the new box
def view_suggested_pos_3d(coords, global_points, bbox_3d):
    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(global_points)
    bbox_3d['center'][0] = coords[0]
    bbox_3d['center'][1] = coords[1]
    linesets = _get_bboxes_wire_frames([bbox_3d], color=(255, 0, 0))

    entities = []
    entities.append(pcd)
    entities.append(linesets[0])
    o3.visualization.draw_geometries(entities)


# plot point cloud from above
def view_suggested_pos_from_above(global_points, coords=[40, 10], old_pos=[0, 0]):
    plt.scatter(global_points[:, 0], global_points[:, 1], s=0.5)
    plt.plot(coords[0], coords[1], color='r', marker='.')
    plt.plot(old_pos[0], old_pos[1], color='black', marker='.')
    plt.axis('scaled')
    plt.axis('off')
    plt.show()


# Adds boxes to the global pointcloud
add_boxes_to_pointcloud(show=False)

# box = update_rotation(box, np.pi)

# Step 1: finding car points inside bbox
# find_points_in_box()

old_pos = box['center']
# Original point cloud without car
# create_point_cloud_without_object(show=False, shift=20, axis=1)
#######################################################################

# Visualize
# new_position = find_position(points, box)
# view_suggested_pos_from_above(points, new_position, old_pos)
# view_suggested_pos_3d(new_position, points, box)


################################################### IMAGE #############################################
# Step 2: cropping the object out of the RGB-image
# # image
# img = plt.imread(file_names_rgb[id])
# img = np.asarray(img)
#
# # label
# label = plt.imread(file_names_2dlabel[id])
# label = np.asarray(label)
# plt.imshow(label)
# plt.axis('off')
# # plt.show()
#
#
# # load colors
# r = np.load('class_colors.npy')


# https://www.python.org/dev/peps/pep-0008/
