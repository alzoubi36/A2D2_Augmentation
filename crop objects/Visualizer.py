from os.path import join
import glob
import numpy as np
import open3d as o3
import json
import matplotlib.pyplot as plt

id = 2

# Lidar Data
root_path = './A2D2-Dataset/Dataset/'
file_names_lidar = sorted(glob.glob(join(root_path, 'Lidar/*.npz')))
print(file_names_lidar)

# 3D Bounding Boxes
file_names_3dbox = sorted(glob.glob(join(root_path, '3DLabel/*.json')))
# print(file_names_3dbox)

# Images
file_names_rgb = sorted(glob.glob(join(root_path, 'RGB/*.png')))
# print(file_names_rgb)

# Labels
file_names_2dlabel = sorted(glob.glob(join(root_path, 'Label/*.png')))
print(file_names_2dlabel)

pointcloud = np.load(file_names_lidar[id])
pc_keys = list(pointcloud.keys())
points = pointcloud['points']
pc_o3 = o3.geometry.PointCloud()
pc_o3.points = o3.utility.Vector3dVector(points)


# o3.visualization.draw_geometries([pc_o3])

def skew_sym_matrix(u):
    return np.array([[0, -u[2], u[1]],
                     [u[2], 0, -u[0]],
                     [-u[1], u[0], 0]])


def axis_angle_to_rotation_mat(axis, angle):
    return np.cos(angle) * np.eye(3) + \
           np.sin(angle) * skew_sym_matrix(axis) + \
           (1 - np.cos(angle)) * np.outer(axis, axis)


def read_bounding_boxes(file_name_bboxes):
    # open the file
    with open(file_name_bboxes, 'r') as f:
        bboxes = json.load(f)

    boxes = []  # a list for containing bounding boxes
    print(bboxes.keys())

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
        boxes.append(bbox_read)

    return boxes


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


bboxes = read_bounding_boxes(file_names_3dbox[id])
box = bboxes[3]
print(get_points(box))

entities_to_draw = []
entities_to_draw.append(pc_o3)

for bbox in bboxes:
    linesets = _get_bboxes_wire_frames([bbox], color=(255, 0, 0))
    entities_to_draw.append(linesets[0])

# o3.visualization.draw_geometries(entities_to_draw)


# Step 1: finding points inside a bbox
ps = get_points(box)
l = np.min(ps, axis=0)
u = np.max(ps, axis=0)
s = np.logical_and(points > l, points < u)
s = np.all(s, axis=1)
car = points[s]
car[:, 1] = car[:, 1]
pointcloud_2 = np.load(file_names_lidar[id + 1])
points_2 = pointcloud_2['points']

car = np.append(car, points_2, axis=0)

# Pointcloud without car
pcd = o3.geometry.PointCloud()
pcd.points = o3.utility.Vector3dVector(points_2)
# o3.visualization.draw_geometries([pcd])

# Pointcloud augmented
pcd = o3.geometry.PointCloud()
pcd.points = o3.utility.Vector3dVector(car)
# o3.visualization.draw_geometries([pcd])


# Step 2: cropping the object out of the RGB-image

# image
img = plt.imread(file_names_rgb[id])
img = np.asarray(img)

# label
label = plt.imread(file_names_2dlabel[id])
label = np.asarray(label)

# find
r = np.load('class_colors.npy')


# indices = np.where(img== r[10])
# d = len(indices[0])
# idex = np.arange(d*2); idex = idex.reshape(d, 2)
# idex[:, 0] = indices[1]
# idex[:, 1] = indices[0]
# plt.imshow(img)
# plt.scatter(idex[:,0], idex[:,1], s=0.2)

# Get object pixels out of Pic
def get_object(r, id):
    x = np.array([0, 0, 0, 0, 0])
    x = x.reshape(1, 5)
    for i in range(len(label)):
        for j in range(len(label[0])):
            # print(j)
            if np.all(label[i][j] == r[id]):
                ids = np.array([i, j, label[i][j][0], label[i][j][1], label[i][j][2]])
                # print(id)
                x = np.append(x, ids.reshape(1, 5), axis=0)
    return x


idex = get_object(r, 14)
# np.save('object_pixels', idex)

# 2D-Box
x = [708, 792, 708, 792]
# y = [1706, 1706, 1963, 1963]
# plt.scatter(y, x)

plt.imshow(label)
# plt.scatter(idex[:,1], idex[:,0], s=0.2)


plt.axis('off')
plt.show()

# put crop on image

# import matplotlib.cm as cm
# im = np.append(label, np.ones(1208*1920).reshape(1208,1920,1),axis=2)
# r = np.append(r, np.ones(19).reshape(19,1),axis=1)
# print(im.shape)
# im = np.where(im == r[15], (r[15][0], r[15][1], r[15][2], r[15][3]), (r[15][0], r[15][1], r[15][2], 0))
# print(im.shape)
# # im = np.ma.masked_where(im == 0.0, im)
# # plt.imshow(img)
# plt.imshow(im)
#
# plt.show()
