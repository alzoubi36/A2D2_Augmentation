import numpy as np
import matplotlib.pyplot as plt
from file_loader import *
from shapely.geometry import Point, Polygon
import cv2
import math as m
from project_utils import *


# Class for instances in the semantic segmentation
class SemanticObject:
    def __init__(self):
        self.bounding_pixel_coords = []
        self.pc_coords = []
        self.class_ = ""
        self.depth = 0.0

    # Bounding Pixels
    def set_bounding_pixels(self, pixels):
        self.bounding_pixel_coords = pixels

    # Pointcloud Points
    def set_pc_coords(self, pc_coords):
        self.pc_coords = pc_coords

    # Class
    def set_class(self, class_):
        self.class_ = class_

    # Depth
    def set_depth(self, depth):
        self.depth = depth


# Takes a 2D semantic seg. image and seperates its labels
class Separator:
    def __init__(self, data):
        self.data = data
        self.semantic_objects = {}

    # Converts HEX colors to normalized RGB colors
    def hex_to_rgb(self, value):
        value = value.lstrip('#')
        lv = len(value)
        rgb = np.asarray(tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)))
        return rgb

    # Finds pixel coordinates of a given class color
    def find_class_pixels(self, color_hex):
        rgb = self.hex_to_rgb(color_hex)
        label = self.data.label * 255.
        image_coords = np.where(np.all(label == rgb, axis=2))
        image_gray = np.where(np.all(label == rgb, axis=2), 1, 0)

        x = image_coords[1]
        y = image_coords[0]
        return x, y, image_gray

    # Separates labels of the same class from each other
    # uses the functions find_contours & find_class_pixels
    def separate_labels(self, class_color_hex):
        class_ = class_list[class_color_hex]
        print(f"Calculating semantic objects of {class_} ...")
        x, y, image_gray = self.find_class_pixels(class_color_hex)
        contours = self.find_contours(image_gray)
        objects = self.create_semantic_objects(contours)
        self.semantic_objects[class_] = objects

    # separates labels of all classes in a semantic segmentation
    def separate_all_labels(self):
        for color in class_list:
            self.separate_labels(color)

    # creates semantic objects <SemanticObject> out of a contour array
    def create_semantic_objects(self, contours):
        objects = []
        for frame in contours:
            sem_object = SemanticObject()

            # bounding pixels
            frame = frame.reshape(len(frame), 2)
            sem_object.set_bounding_pixels(frame)

            # pc coords
            points = self.find_instance_pc_points(sem_object)
            sem_object.set_pc_coords(points)

            # depth values
            average_depth = self.calculate_object_depth_value(sem_object)
            sem_object.set_depth(average_depth)

            objects.append(sem_object)
        return objects

    # Finds grouped contours in the sem. segmentation
    def find_contours(self, image):
        thresh = cv2.threshold(image.astype(np.uint8) * 255,
                               30, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        return contours

    # Finds pc pixels of an instance in the semantic segmentation
    # Instance type: <SemanticObject>
    def find_instance_pc_points(self, sem_object):
        try:
            points = []
            frame = Polygon(sem_object.bounding_pixel_coords)
            shapely_points = self.convert_pc_pixels_to_shapely(self.data.pointcloud)
            point_ids = []
            for i in range(len(shapely_points)):
                if frame.contains(shapely_points[i]):
                    point_ids.append(i)
                    points.append(self.data.pointcloud['points'][i])
            return points
        except:
            return None

    # Converts points of a PC to type <Point> of the Shapely library
    # Necessary to check if point is inside a contour using Polygon.contains(Point)
    def convert_pc_pixels_to_shapely(self, pointcloud):
        x = pointcloud['col']
        y = pointcloud['row']
        points = []
        for i in range(len(x)):
            points.append(Point(x[i], y[i]))
        return points

    # Returns the distance of a given point from front_center_camera
    def calculate_depth_value(self, point):
        x = point[0]
        y = point[1]
        z = point[2]
        return m.sqrt(x ** 2 + y ** 2 + z ** 2)

    # calculates the average depth of a semantic object <SemanticObject>
    def calculate_object_depth_value(self, sem_object):
        try:
            depth_values = [self.calculate_depth_value(i) for i in sem_object.pc_coords]
            average_depth = sum(depth_values) / len(depth_values)
        except:
            average_depth = None
        return average_depth

    # Plots a given class by its color
    def plot_data(self, color):
        label = self.data.label
        x, y, image_gray = self.find_class_pixels(color)
        plt.imshow(self.data.image)
        plt.scatter(x, y, s=0.1, color="b")
        plt.axis("off")
        plt.show()

# data = Loader('./A2D2-Dataset/Dataset/', 8)
# #
# sep = Separator(data)
#
# id = "#b97a57"
# sep.seperate_labels(id)
# frame = sep.semantic_objects['Road blocks'][0].bounding_pixel_coords
# plt.plot(frame[:, 0], frame[:, 1])
# sep.plot_data(id)
# print(len(sep.semantic_objects['Road blocks']))
