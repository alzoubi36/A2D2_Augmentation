from file_loader import *
from shapely.geometry import Point, Polygon
import cv2
import math as m


class SemanticObject:
    """
    Class for instances in the semantic segmentation
    """
    def __init__(self):
        self.bounding_pixel_coords = []
        self.pixels = []
        self.pc_coords = []
        self.class_ = ""
        self.depth = 0.0

    # Bounding Pixels
    def set_bounding_pixels(self, pixels):
        self.bounding_pixel_coords = pixels

    # pixels
    def set_pixels(self, x, y):
        self.pixels = np.append(x.reshape(x.shape[0], 1),
                                y.reshape(y.shape[0], 1), axis=1)

    # Pointcloud Points
    def set_pc_coords(self, pc_coords):
        self.pc_coords = pc_coords

    # Class
    def set_class(self, class_):
        self.class_ = class_

    # Depth
    def set_depth(self, depth):
        self.depth = depth


class Separator:
    """
    Takes a 2D semantic seg. image and seperates its labels
    """
    def __init__(self, data):
        self.data = data
        self.semantic_objects = {}

    def hex_to_rgb(self, value):
        """
        Converts HEX colors to normalized RGB colors
        :param value: hexadecimal color <str>
        :return: rgb color [r, g, b]
        """
        value = value.lstrip('#')
        lv = len(value)
        rgb = np.asarray(tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)))
        return rgb

    def find_class_pixels(self, color_hex, undistort=False):
        """
        Finds pixel coordinates of a given class color
        :param color_hex: <str>
        :param undistort: pixels associated to the undistort label?
        :return: class pixels [x], [y], mask where class is white
        """
        rgb = self.hex_to_rgb(color_hex)
        label = self.data.label
        if undistort:
            label = undistort_image(label, 'front_center', label=True)
        label = self.data.label * 255
        image_coords = np.where(np.all(label[:, :, :3] == rgb, axis=2))
        mask = np.where(np.all(label[:, :, :3] == rgb, axis=2), 1, 0)

        x = image_coords[1]
        y = image_coords[0]
        return x, y, mask

    def separate_labels(self, class_color_hex):
        """
        Separates labels of the same class from each other
        uses the functions find_contours & find_class_pixels
        :param class_color_hex: <str>
        :return: None
        """
        class_ = class_list[class_color_hex]
        print(f"Calculating semantic objects of {class_} ...")
        x, y, image_gray = self.find_class_pixels(class_color_hex)
        contours = self.find_contours(image_gray)
        objects = self.create_semantic_objects(contours, x, y)
        self.semantic_objects[class_] = objects

    def separate_all_labels(self):
        """
        separates labels of all classes in a semantic segmentation
        :return: None
        """
        for color in class_list:
            self.separate_labels(color)

    def create_semantic_objects(self, contours, x, y):
        """
        creates semantic objects <SemanticObject> out of a contour array
        :param contours: class contours
        :param x: pixel-x
        :param y: pixel-y
        :return: <SemanticObject> objects
        """
        objects = []
        for frame in contours:
            sem_object = SemanticObject()

            # bounding pixels
            frame = frame.reshape(len(frame), 2)
            sem_object.set_bounding_pixels(frame)

            # pixels
            # try:
            #     x_sem, y_sem = self.find_pixels_in_contour(sem_object, x, y)
            #     sem_object.set_pixels(x_sem, y_sem)
            # except:
            #     pass

            # pc coords
            points = self.find_instance_pc_points(sem_object)
            sem_object.set_pc_coords(points)

            # depth values
            average_depth = self.calculate_object_depth_value(sem_object)
            sem_object.set_depth(average_depth)

            objects.append(sem_object)
        return objects

    def find_contours(self, image):
        """
        Finds grouped contours in the sem. segmentation
        :param image: <ndarray>
        :return: contours <ndarray>
        """
        thresh = cv2.threshold(image.astype(np.uint8) * 255,
                               30, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        return contours

    def find_pixels_in_contour(self, sem_object, x, y):
        """
        finds pixels of a semantic object given its contours
        Instance type: <SemanticObject>
        :param sem_object: <SemanticObject>
        :param x: x-pixels
        :param y: y-pixels
        :return: pixels inside contour [x, y]
        """
        try:
            points = []
            frame = Polygon(sem_object.bounding_pixel_coords)
            shapely_points = self.convert_pc_pixels_to_shapely(x, y)
            for i in range(len(shapely_points)):
                if frame.contains(shapely_points[i]):
                    points.append((x[i], y[i]))
            points = np.array(points)
            x = points[:, 0]
            y = points[:, 1]
            return x, y
        except:
            return

    def find_instance_pc_points(self, sem_object):
        """
        Finds pc pixels of an instance in the semantic segmentation
        Instance type: <SemanticObject>
        :param sem_object: <SemanticObject>
        :return: pc-points <n x 3> <ndarray>
        """
        try:
            points = []
            frame = Polygon(sem_object.bounding_pixel_coords)
            x = self.data.pointcloud['col']
            y = self.data.pointcloud['row']
            shapely_points = self.convert_pc_pixels_to_shapely(x, y)
            point_ids = []
            for i in range(len(shapely_points)):
                if frame.contains(shapely_points[i]):
                    point_ids.append(i)
                    points.append(self.data.pointcloud['points'][i])
            return points
        except:
            return

    def convert_pc_pixels_to_shapely(self, x, y):
        """
        Converts points of a PC to type <Point> of the Shapely library
        Necessary to check if point is inside a contour using Polygon.contains(Point)
        :param x: x-pixels
        :param y: y-pixels
        :return: shapley points
        """
        points = []
        for i in range(len(x)):
            points.append(Point(x[i], y[i]))
        return points

    def calculate_depth_value(self, point):
        """
        Returns the distance of a given point from front_center_camera
        :param point: 3d-point [x, y, z]
        :return: depth <float>
        """
        x = point[0]
        y = point[1]
        z = point[2]
        return m.sqrt(x ** 2 + y ** 2 + z ** 2)

    def calculate_object_depth_value(self, sem_object):
        """
        calculates the average depth of a semantic object <SemanticObject>
        :param sem_object: <SemanticObject>
        :return: average depth <float>
        """
        try:
            depth_values = [self.calculate_depth_value(i) for i in sem_object.pc_coords]
            average_depth = sum(depth_values) / len(depth_values)
        except:
            average_depth = None
        return average_depth

    def plot_data(self, color):
        """
        Plots a given class by its color
        :param color: hexadecimal class color <str>
        :return: None
        """
        label = self.data.label
        x, y, image_gray = self.find_class_pixels(color)
        plt.imshow(self.data.image)
        plt.scatter(x, y, s=0.1, color="b")
        plt.axis("off")
        plt.show()

    def create_specified_mask(self, car=True, pedestrian=True, bicycle=False):
        """
        Generates custom masks for the classes: Car, Pedestrian and Bicycle
        :param car: should cars be seen?
        :param pedestrian: should pedestrians be seen?
        :param bicycle: should bicycles be seen?
        :return: mask
        """
        x, y, mask = self.find_class_pixels('C0C0C0')
        classes = {'Car':car, 'Pedestrian':pedestrian , 'Bicycle':bicycle}
        sem_classes = list(class_list.values())
        for class_, allowed in classes.items():
            if allowed:
                for i in range(len(class_list)):
                    if class_ in sem_classes[i]:
                        color = list(class_list.keys())[i]
                        x, y, mask_c = self.find_class_pixels(color)
                        mask += mask_c
        return mask

# data = Loader(path, 550)
# # #
# sep = Separator(data)
# sep.create_specified_mask(car=True)
# id = "#f1e6ff"
# sep.separate_labels(id)
# # sep.separate_all_labels()
# frame = sep.semantic_objects[class_list[id]][0].bounding_pixel_coords
# # plt.plot(frame[:, 0], frame[:, 1])
# sep.plot_data(id)
# print(len(sep.semantic_objects[class_list[id]]))
