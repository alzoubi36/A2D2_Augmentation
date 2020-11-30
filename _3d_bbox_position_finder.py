from pc_to_image_projector import *
from file_loader import *

data = Loader(path, 6)

# box = Loader('./A2D2-Dataset/Dataset/', 11).boxes[5]
# pointcloud = data.pointcloud
# points = pointcloud['points']
# label = data.label
# image = data.image


# calculate the rotation angle of a new position
def new_position_rotation_angle(box, new_pos, old_pos):
    old_pos = old_pos[:2]
    # print("old: ", old_pos)
    # print("new: ", new_pos)
    angle_cos = np.dot(old_pos, new_pos) / \
                (np.linalg.norm(old_pos) * np.linalg.norm(new_pos))
    angle = np.arccos(angle_cos)
    # print("angle: ", angle * 180 / np.pi)
    if old_pos[1] < new_pos[1]:
        return angle
    else:
        return -angle


# propose Positions for the new point cloud
def generate_positions(points):
    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])
    rand_x = np.random.randint(x_min + 2, x_max - 80)
    rand_y = np.random.randint(y_min + 15, y_max - 15)
    # rand_y = np.random.randint(old_pos[1], - old_pos[1])
    return rand_x, rand_y


# checks proposed Positions. Are there 3d-points inside the 3d box?
# takes size and ground tolerance refine results.
def check_positions(coords, box, points, old_pos, ground_tolerance=0.11, size_tolerance=0.5):
    box_3d = box.copy()
    box_3d['center'] = box['center'].copy()
    angle = new_position_rotation_angle(box, coords, old_pos)
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
def find_position(global_points, bbox_3d, old_pos):
    bad_position = True
    while bad_position:
        new_position = generate_positions(global_points)
        bad_position = not check_positions(new_position, bbox_3d, global_points, old_pos)

    return new_position


# plot pointcloud with the new box
def view_suggested_pos_3d(coords, global_points, bbox_3d):
    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(global_points)
    bbox_3d['center'][0] = coords[0]
    bbox_3d['center'][1] = coords[1]
    linesets = get_bboxes_wire_frames([bbox_3d], color=(255, 0, 0))

    entities = [pcd, linesets[0]]
    o3.visualization.draw_geometries(entities)


# plot point cloud from above
def view_suggested_pos_from_above(global_points, coords=[40, 10], old_pos=[0, 0]):
    plt.scatter(global_points[:, 0], global_points[:, 1], s=0.5)
    plt.plot(coords[0], coords[1], color='r', marker='.')
    plt.plot(old_pos[0], old_pos[1], color='black', marker='.')
    plt.axis('scaled')
    plt.axis('off')
    plt.show()


##### Testing Area #####

# data_box = Loader('./A2D2-Dataset/Dataset/', 11)
# old_pos = data_box.boxes[5]['center']
#
#
# # Visualize
# new_position = find_position(points, box, old_pos)
# # old_pos = box['center']
# view_suggested_pos_from_above(points, new_position, old_pos)
# view_suggested_pos_3d(new_position, points, box)
#
# # show boxes
# new_box = box.copy()
# new_box['center'][0] = new_position[0]
# new_box['center'][1] = new_position[1]
# draw_box(get_points(box), image)
