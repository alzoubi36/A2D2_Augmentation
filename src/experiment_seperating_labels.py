import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from skimage.filters import sobel
from skimage.color import rgb2gray
from file_loader import *
from shapely.geometry import Point, Polygon
from semantic_partitioner import *


# finds pixel coordinates of a given color
def find_color_pixels(color_hex, label):
    rgb = hex_to_rgb(color_hex)
    label = label * 255.
    image_coords = np.where(np.all(label == rgb, axis=2))
    image = np.where(np.all(label == rgb, axis=2), 1, 0)
    x = image_coords[1]
    y = image_coords[0]
    return x, y, image


def findContours(img, threshold):
    contours = measure.find_contours(img, threshold)
    return contours


def findContours_cv2(gray):
    thresh = cv2.threshold(gray.astype(np.uint8) * 255,
                           30, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    rgb = np.asarray(tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)))
    return rgb


def createOnecont(contours):
    contours.sort(key=len)
    reduced = contours
    onecont = np.ones((1, 2))
    # merge
    for i in reduced:
        onecont = np.append(onecont, i, axis=0)
    onecont = onecont[1:]

    return onecont


def concatenate_pixel_coords(x, y):
    x = x.reshape(x.shape[0], 1)
    y = y.reshape(y.shape[0], 1)
    pixels = np.append(y, x, axis=1)
    return pixels


def create_shapely_points(pointcloud):
    x = pointcloud['col']
    y = pointcloud['row']
    points = []
    for i in range(len(x)):
        points.append(Point(x[i], y[i]))
    return points


def find_pc_points(sem_obj):
    frame = Polygon(sem_obj.bounding_pixel_coords)
    points = create_shapely_points(data.pointcloud)
    print(len(points))
    point_ids = []
    for i in range(len(points)):
        if frame.contains(points[i]):
            point_ids.append(i)
    return point_ids


def show_skimage_contours():
    contours = findContours(image, 0.2)
    print(len(contours))
    contours = createOnecont(contours)
    plt.imshow(image)
    plt.imshow(image, alpha=0.5)
    plt.scatter(contours[:, 1], contours[:, 0], s=10, color="r")
    plt.axis("off")
    plt.show()


def show_opencv_contours():
    contours_cv2 = findContours_cv2(image)
    print(len(contours_cv2))
    plt.imshow(image)
    plt.imshow(data.image, alpha=0.5)
    for i in contours_cv2:
        i = i.reshape(len(i), 2)
        plt.plot(i[:, 0], i[:, 1])
    plt.axis("off")
    plt.show()

# ------------------------------------------
#       Countours Tests
# ------------------------------------------
# data = Loader('./A2D2-Dataset/Dataset/', 8)
# label = data.label
# x, y, image = find_color_pixels("#ff00ff", label)
# show_skimage_contours()
#show_opencv_contours()
# ------------------------------------------




# ------------------------------------------
#       Depth Value Tests
# ------------------------------------------
data = Loader('./A2D2-Dataset/Dataset/', 12)

sep = Separator(data)

id = "#ff0000"
id_obj = 7
sep.seperate_labels(id)
print(sep.semantic_objects[class_list[id]])
# sem_object2 = sep.semantic_objects[class_list[id]][id_obj]
# frame = sep.semantic_objects[class_list[id]][id_obj].bounding_pixel_coords
# plt.plot(frame[:, 0], frame[:, 1], color='r')

try:
    for sem_object in sep.semantic_objects[class_list[id]]:
        frame = sem_object.bounding_pixel_coords
        plt.plot(frame[:, 0], frame[:, 1], color='r')
        ids = find_pc_points(sem_object)
        for i in ids:
            plt.scatter(data.pointcloud['col'][i], data.pointcloud['row'][i], s=0.1, color='b')
except:
    pass

#sep.plot_data(id)
plt.imshow(data.image)
plt.axis('off')
plt.show()
print(len(sep.semantic_objects[class_list[id]]))


