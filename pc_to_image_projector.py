from file_loader import *


# Project into undistorted image
def project_from_pc_to_image_undist(point):
    point = np.array([point[1], point[2], point[0]])
    image_coords = np.dot(cam_matrix_undist, point)
    image_coords = image_coords / image_coords[2]
    return image_coords[0], image_coords[1]


# Project into distorted image
def project_from_pc_to_image_dist(point):
    point = np.array([point[1], point[2], point[0]])
    rtemp = ttemp = np.array([0, 0, 0], dtype='float32')
    dist_parms = np.asarray([[-0.2611312587700434, 0.0, 0.0, 0.0, 0.0]])
    output = cv2.projectPoints(point, rtemp, ttemp, cam_matrix_dist, dist_parms)
    x = output[0][0][0][0]
    y = output[0][0][0][1]
    return x, y


# General projection function from PC to image
def project_to_image(point, undist=True):
    if undist:
        point = project_from_pc_to_image_undist(point)
    else:
        point = project_from_pc_to_image_dist(point)
    return point


# creates line_sets for drawing a bbox in image space
def project_box_from_pc_to_image(points, undist=True):
    pixels = np.array([project_to_image(i, undist) for i in points])
    line_set = np.asarray([pixels[0], pixels[1], pixels[2],
                           pixels[3], pixels[0],
                           pixels[4], pixels[1], pixels[5],
                           pixels[2], pixels[6], pixels[3],
                           pixels[7], pixels[4], pixels[5],
                           pixels[6], pixels[7], pixels[4],
                           pixels[5], pixels[2], pixels[1],
                           pixels[6]])
    return line_set


# draws a 3d-bbox in a rgb-image
def draw_box(points, image, undist=True):
    line_set = project_box_from_pc_to_image(points, undist)
    plt.plot(line_set[:, 0], line_set[:, 1], linewidth=0.5, color="r")
    plt.imshow(image)
    plt.axis("off")
    # plt.show()

# -------------------------------
#       error reproduction
# -------------------------------
# data = Loader(path, 20)
# id = 0
# box = data.boxes[id]
# points = get_points(box)
# # print(data.boxes[id]['class'])
# # print(box)
# image = data.image
# draw_box(points, image, undist=False)
# # image = undistort_image(image, 'front_center')
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
