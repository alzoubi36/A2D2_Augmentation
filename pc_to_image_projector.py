from file_loader import *


def project_from_pc_to_image_undist(point):
    """
    Projects into undistorted image
    :param point: [x, y, z]
    :return: [u, v]
    """
    point = np.array([point[1], point[2], point[0]])
    image_coords = np.dot(cam_matrix_undist, point)
    image_coords = image_coords / image_coords[2]
    return image_coords[0], image_coords[1]


def project_from_pc_to_image_dist(point):
    """
    Projects into distorted image
    :param point: [x, y, z]
    :return: [u, v]
    """
    point = np.array([point[1], point[2], point[0]])
    rtemp = ttemp = np.array([0, 0, 0], dtype='float32')
    dist_parms = np.asarray([[-0.2611312587700434, 0.0, 0.0, 0.0, 0.0]])
    output = cv2.projectPoints(point, rtemp, ttemp, cam_matrix_dist, dist_parms)
    x = output[0][0][0][0]
    y = output[0][0][0][1]
    return x, y


def project_to_image(point, undist=True):
    """
    General projection function from PC to image
    :param point: [x, y, z]
    :param undist: to distorted or undistorted?
    :return: [u, v]
    """
    if undist:
        point = project_from_pc_to_image_undist(point)
    else:
        point = project_from_pc_to_image_dist(point)
    return point


def project_box_from_pc_to_image(points, undist=True):
    """
    creates line_sets for drawing a bbox in image space
    :param points: box points <8 x 3>
    :param undist: to distorted or undistorted?
    :return: line set <20 x [u, v]>
    """
    pixels = np.array([project_to_image(i, undist) for i in points])
    line_set = np.asarray([pixels[0], pixels[1], pixels[2],
                           pixels[3], pixels[0],
                           pixels[4], pixels[1], pixels[5],
                           pixels[2], pixels[6], pixels[3],
                           pixels[7], pixels[4], pixels[5],
                           pixels[6], pixels[7], pixels[4],
                           pixels[5], pixels[2], pixels[1],
                           pixels[6]]).astype('float32')
    return line_set


def draw_box(points, image, undist=True):
    """
    draws a 3d-bbox in a rgb-image
    :param points: box points <8 x 3>
    :param undist: to distorted or undistorted?
    :param image: as loaded with Loader <image>
    :return: None
    """
    line_set = project_box_from_pc_to_image(points, undist)
    plt.plot(line_set[:, 0], line_set[:, 1], linewidth=0.5, color="b")
    plt.imshow(image)
    plt.axis("off")

# -------------------------------
#       error reproduction
# -------------------------------
# data = Loader(path, 21)
# id = 2
# box = data.boxes[id]
# points = get_points(box)
# print(data.pointcloud['row'][0])
# print(data.pointcloud['col'][0])
# print(project_to_image(data.pointcloud['points'][0], undist=True))
# # print(data.boxes[id]['class'])
# # print(box)
# image = data.image
# image = undistort_image(image, 'front_center')
# draw_box(points, image, undist=True)
# # x = [box['left'], box['left'], box['right'], box['right'], box['left']]
# # y = [box['bottom'], box['top'], box['top'], box['bottom'], box['bottom']]
# # plt.plot(y, x)
# plt.imshow(image)
# plt.axis('off')
# plt.show()


##### Testing Area #####
# #
# data = Loader(path, 116)
# pointcloud = data.pointcloud
# # # for i in range(0, len(data.boxes)):
# # points = get_points(data.boxes[5])
# # print(data.boxes[5]['class'])
# image = data.image
# # draw_box(points, image)
# # y = pointcloud['row']
# # x = pointcloud['col']
# pts = np.array([project_from_pc_to_image(cam_matrix_undist, i) for i in pointcloud['points']])
# # # print(pts.shape)
# # image = undistort_image(image, 'front_center')
# label = undistort_image(data.label, 'front_center')
# # plt.scatter(pts[:, 1], pts[:, 0], s=0.1, color='y')
# # # plt.scatter(x, y, s=0.1, color='b')
# # plt.imshow(image)
# img = map_lidar_points_onto_image(image, pointcloud)
# # plt.imshow(label, alpha=0.5)
# plt.imshow(img)
# #
# # plt.axis('off')
# plt.show()
#
# points = pointcloud['points']
# pc_o3 = o3.geometry.PointCloud()
# pc_o3.points = o3.utility.Vector3dVector(points)
# # entities_to_draw = [pc_o3]
# entities_to_draw = [pc_o3]
# for bbox in data.boxes:
#     linesets = get_bboxes_wire_frames([bbox], color=(255, 0, 0))
#     entities_to_draw.append(linesets[0])
#     break
# o3.visualization.draw_geometries(entities_to_draw)
# #
