from pc_to_image_projector import *
from file_loader import *


def new_position_rotation_angle(box, new_pos, old_pos):
    """
    calculates the rotation angle of a new position in pc
    :param box: as loaded with Loader <json>
    :param new_pos: [x, y]
    :param old_pos: [x, y]
    :return: angle in radians
    """
    old_pos = old_pos[:2]
    new_pos = np.asarray(new_pos).astype('float64')
    angle = np.math.atan2(np.linalg.det([new_pos[:2], old_pos[:2]]),
                          np.dot(new_pos[:2], old_pos[:2]))
    return angle


def generate_positions(points):
    """
    propose Positions in the new point cloud
    :param points: pc-points <n x 3>
    :return: position [x, y]
    """
    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])
    rand_x = np.random.randint(x_min + 10 , x_max-40)
    rand_y = np.random.randint(y_min + 10, y_max - 10)
    # rand_y = np.random.randint(old_pos[1], - old_pos[1])
    return rand_x, rand_y


def check_positions(coords, box_3d_, points, old_pos, old_angle,
                    ground_tolerance=0.11, size_tolerance=0.5):
    """
    checks proposed Positions. Are there 3d-points inside the 3d box?
    takes size and ground tolerance refine results.
    :param coords: new position
    :param box: as loaded with Loader <json>
    :param points: pc-points <n x 3>
    :param old_pos: [x, y]
    :param old_angle: radians
    :param ground_tolerance: how much of ground fluctuations should be ignored?
    :param size_tolerance: tolerance of the axes parallel 2d-box
    :return: True if good position else False
    """
    box_3d = box_3d_.copy()
    box_3d['center'] = box_3d_['center'].copy()
    angle = new_position_rotation_angle(box_3d_, coords, old_pos)
    box_3d['center'][0] = coords[0]
    box_3d['center'][1] = coords[1]
    coords = np.asarray(coords)
    update_rotation(box_3d, angle, old_angle)

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


def find_position(global_points, bbox_3d, old_pos, old_angle):
    """
    Find a good Position
    :param global_points: global pc <n x 3>
    :param bbox_3d: as loaded with Loader <json>
    :param old_pos: [x, y]
    :param old_angle: radians
    :return: a good checked position
    """
    bad_position = True
    while bad_position:
        new_position = generate_positions(global_points)
        bad_position = not check_positions(new_position, bbox_3d, global_points, old_pos, old_angle)

    return new_position


def view_suggested_pos_3d(coords, global_points, bbox_3d):
    """
    plot pointcloud and the new box in 3d space
    :param coords: new position
    :param global_points: pc points <3 x n>
    :param bbox_3d: as loaded with Loader <json>
    :return: None
    """
    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(global_points)
    bbox_3d['center'][0] = coords[0]
    bbox_3d['center'][1] = coords[1]
    linesets = get_bboxes_wire_frames([bbox_3d], color=(255, 0, 0))

    entities = [pcd, linesets[0]]
    o3.visualization.draw_geometries(entities)


def view_suggested_pos_from_above(global_points, coords=[40, 10], old_pos=[0, 0]):
    """
    plot point cloud from above
    :param coords: new position
    :param global_points: pc points <3 x n>
    :param old_pos: [x, y]
    :return: None
    """
    plt.scatter(global_points[:, 0], global_points[:, 1], s=0.5)
    plt.plot(coords[0], coords[1], color='r', marker='.')
    plt.plot(old_pos[0], old_pos[1], color='black', marker='.')
    plt.axis('scaled')
    plt.axis('off')
    plt.show()


##### Testing Area #####

# data = Loader(path, 469)
# old_pos = data.boxes[0]['center']
# box=data.boxes[0]
# points = data.pointcloud['points']
# new_position = old_pos
# # Visualize
# # new_position = find_position(points, box, old_pos)
# new_position = old_pos
# old_pos = box['center']
# view_suggested_pos_from_above(points, new_position, old_pos)
# view_suggested_pos_3d(new_position, points, box)

# show boxes
# image = data.image
# # image = undistort_image(image, 'front_center')
# new_box = box.copy()
# new_box['center'][0] = new_position[0]
# new_box['center'][1] = new_position[1]
# draw_box(get_points(box), image)
# plt.show()