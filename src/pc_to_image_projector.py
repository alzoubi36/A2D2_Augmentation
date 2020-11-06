import json
import numpy as np
from file_loader import *
from project_utils import *


# Projection functions. Both deliver the same results.
def project_from_pc_to_image(cam_matrix, point):
    point = np.array([point[1], point[2], point[0]])
    image_coords = np.dot(cam_matrix, point)
    image_coords = image_coords / image_coords[2]

    return image_coords[1], image_coords[0]


def project_from_pc_to_image_alternative(cam_matrix, point):
    point = np.array([point[1], point[2], point[0]])
    pix_y = (point[0] * cam_matrix[0][0]) / point[2] + cam_matrix[0][2]
    pix_x = (point[1] * cam_matrix[1][1]) / point[2] + cam_matrix[1][2]
    return pix_x, pix_y


# creates line_sets for drawing a bbox in image space
def project_box_from_pc_to_image(points, cam_matrix=cam_matrix_undist):
    pixels = np.array([project_from_pc_to_image(cam_matrix, i) for i in points])
    line_set = np.asarray([pixels[0], pixels[1], pixels[2],
                           pixels[3], pixels[0],
                           pixels[4], pixels[1], pixels[5],
                           pixels[2], pixels[6], pixels[3],
                           pixels[7], pixels[4], pixels[5],
                           pixels[6], pixels[7], pixels[4],
                           pixels[5], pixels[2], pixels[1],
                           pixels[6]])
    return line_set


# draws a 3d-bbox in an rgb-image
def draw_box(points, image):
    line_set = project_box_from_pc_to_image(points, cam_matrix_undist)
    plt.imshow(image)
    plt.plot(line_set[:, 1], line_set[:, 0], linewidth=0.5, color="r")
    plt.axis("off")
    plt.show()


##### Testing Area #####

# data = Loader('./A2D2-Dataset/Dataset/', 3)
# pointcloud = data.pointcloud
# # for i in range(2, len(data.boxes)):
# points = get_points(data.boxes[9])
# print(points)
# image = data.image
# draw_box(points, image)
# y = pointcloud['row']
# x = pointcloud['col']
# pts = np.array([project_from_pc_to_image(cam_matrix_undist, i) for i in pointcloud['points']])
# print(pts.shape)
# image = undistort_image(image, 'front_center')
# plt.scatter(pts[:, 1], pts[:, 0], s=0.1)
# plt.scatter(x, y, s=0.1, color='b')
# plt.imshow(image)
# plt.axis('off')
# plt.show()
