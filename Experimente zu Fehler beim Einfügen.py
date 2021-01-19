from file_loader import *
from pc_to_image_projector import *

# def unit_vector(vector):
#     """ Returns the unit vector of the vector.  """
#     return vector / np.linalg.norm(vector)
#
# def angle_between(v1, v2):
#     """ Returns the angle in radians between vectors 'v1' and 'v2'::
#     """
#     v1_u = unit_vector(v1)
#     v2_u = unit_vector(v2)
#     return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# changes the rotation matrix of a 3d bbox according to a rotation angle
def update_rotation(box, angle, old_angle):
    # axis = box['axis']
    axis = np.array([0, 0, 1]).astype('float32')
    print('old angle: ', np.degrees(old_angle))
    if axis[2] > 0:
        angle = old_angle - angle
    else:
        angle = old_angle + angle
    box["rotation"] = axis_angle_to_rotation_mat(axis, angle)
    box['angle'] = angle
    print("new angle: ", angle * 180 / np.pi)
    return box


# calculate the rotation angle of a new position
def new_position_rotation_angle(new_pos, old_pos):
    # old_pos = old_pos[:2]
    print('old pos', old_pos)
    # print("old: ", old_pos)
    print("new pos: ", new_pos)
    # angle_cos = np.dot(old_pos, new_pos) / \
    #             (np.linalg.norm(old_pos) * np.linalg.norm(new_pos))
    # angle = np.arccos(angle_cos)
    # angle = angle_between(new_pos, old_pos)

    angle = np.math.atan2(np.linalg.det([new_pos[:2], old_pos[:2]]),
                          np.dot(new_pos[:2], old_pos[:2]))
    if old_pos[1] < new_pos[1]:
        return -angle
    else:
        return angle



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
        points1 = np.asarray([brl, bfl, bfr, brr, trl, tfl, tfr, trr])
        points = np.dot(points, bbox['rotation'].T)

        # add center position
        points = points + bbox['center']
    # print('with center', points[0]+bbox['center'])
    # print('without center', points[0])

    return points, points1

def get_data():
    for i in range(0, 100):
        data = Loader(path, i)
        # name = data.file_names_lidar[i].split('\\')[-1][:-4]
        name = '20180807145028_lidar_frontcenter_000009789_0.png'
        # for j in os.listdir('./foregrounds'):
        #     if name in j:
        #         print(f'{i}: ', True)
        if name in os.listdir('./foregrounds'):
            print(f'{i}: ', True)
            return data

def get_forground_id(forground):
    data = Loader(path, 0)
    for i, name in enumerate(data.file_names_lidar):
        if name.split('\\')[-1][:-4] == forground[:-6]:
            print('id: ', i)
            return i


# data = get_data()
# box = data.boxes[0]
#
# data2 = get_data()
# box2 = data2.boxes[0]
#
# box2['center'][1] = -box2['center'][1]
# angle = new_position_rotation_angle(box2['center'], box['center'])
# update_rotation(box2, angle, box['angle'])
#
# pixels = project_box_from_pc_to_image(get_points(box), True)
# max1 = np.max(pixels, axis=0)
# min1 = np.min(pixels, axis=0)
#
# pixels2 = project_box_from_pc_to_image(get_points(box2), True)
# max2 = np.max(pixels2, axis=0)
# min2 = np.min(pixels2, axis=0)

# id = 1
# print('result 1:', max1[id] - min1[id])
# print('result 2:', max2[id] - min2[id])
#





#--------------- Test 2 - Why are the new distances not the same? ---------------
# get_forground_id('20180807145028_lidar_frontcenter_000000544_0.png')

id = get_forground_id('20180807145028_lidar_frontcenter_000002031_0.png')
data = Loader(path, id)
plt.imshow(data.image)
# plt.show()
# box = data.boxes[0]
# data2 = Loader(path, id)
# old_angle = data2.boxes[0]['angle']
# # box['center'][1] = -box['center'][1]
# # box['center'][0] = box['center'][1]
# # box['angle'] = -np.pi/4
# # box['rotation'] = axis_angle_to_rotation_mat(box['axis'], box['angle'])
# new_pos = [-box['center'][0], box['center'][1], box['center'][2]]
# # axis = np.array([0, 0, 1])
#
# id = 3
# points, p1 = get_points(box)
# magnitude = np.linalg.norm(points[id])
# magnitude_orig = np.linalg.norm(p1[id])
# pixels = project_box_from_pc_to_image(points, undist=True)
# max_1 = np.max(pixels, axis=0)
# min_1 = np.min(pixels, axis=0)
# print('max1 - min1: ', max_1 - min_1)
#
# # plt.scatter(box['center'][0], box['center'][1])
# plt.plot([0, box['center'][0]], [0, box['center'][1]])
# print('magnitude:', magnitude)
#
#
# angle = new_position_rotation_angle(new_pos, box['center'])
# box['center'] = new_pos
# box = update_rotation(box, angle, old_angle)
# points_new, p1_new = get_points(box)
# pixels = project_box_from_pc_to_image(points_new, undist=True)
# max_1 = np.max(pixels, axis=0)
# min_1 = np.min(pixels, axis=0)
# print('max2 - min2: ', max_1 - min_1)
#
# magnitude_new = np.linalg.norm(points_new[id])
# magnitude_new_orig = np.linalg.norm(p1_new[id])
# plt.scatter(points[0][0], points[0][1])
# plt.scatter(points_new[0][0], points_new[0][1])
# # plt.scatter(box['center'][0], box['center'][1])
# plt.plot([0, box['center'][0]], [0, box['center'][1]])
# plt.scatter(0, 0)
#
# # print('points_new: ', points_new)
# # print('points: ', points)
# print('new magnitude:', magnitude_new)
#
# plt.plot(points[:, 0], points[:, 1])
# plt.plot(points_new[:, 0], points_new[:, 1])
#
#
# # plt.axis('scaled')
# plt.axis('square')
# plt.show()

# # --------------- Test 1 - around which axis do we rotate? -----------------
# p1 = np.array([20, 20, 0])
# p2 = np.array([10, 5, 0])
# pts = np.append(p1.reshape(1, 3), p2.reshape(1, 3), axis=0)
# plt.scatter(pts[:, 0], pts[:, 1])
# plt.plot(pts[:, 0], pts[:, 1])
# axis = np.array([0, 0, 1])
# angle = np.radians(30)
# rot = axis_angle_to_rotation_mat(axis, angle)
# pts_rot = np.dot(pts, rot.T)
# plt.scatter(pts_rot[:, 0], pts_rot[:, 1], color='r')
# plt.plot(pts_rot[:, 0], pts_rot[:, 1], color='r')
# plt.plot(0, 0)
# id = 0
# x = plt.axis('scaled')
# x2 = plt.axis('square')
# print('original:', pts[id])
# print('magnitude:', np.linalg.norm(pts[id]))
# print('rotated:', pts_rot[id])
# print('magnitude:', np.linalg.norm(pts_rot[id]))
# print(pts.shape)
#
# plt.show()

