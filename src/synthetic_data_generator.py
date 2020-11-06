from project_utils import *
from _3d_bbox_position_finder import *
from semantic_partitioner import *

data = Loader('./A2D2-Dataset/Dataset/', 0)


# Class for generating synthetic data for the A2D2-Dataset.
class SyntheticDataGenerator:
    def __init__(self, data):
        self.pointcloud = data.pointcloud
        self.points = self.pointcloud['points']
        self.label = data.label
        self.get_semantic_objects()

        # Todo: create objects for augmentation (boxes, labels and crops)
        box = data.boxes[1]

    def propose_3dbbox_position(self):
        return find_position(self.points, box)

    def project_3dbox_into_image(self):
        new_position = self.propose_3dbbox_position()
        new_box = box.copy()
        new_box['center'][0] = new_position[0]
        new_box['center'][1] = new_position[1]
        pixels = project_box_from_pc_to_image(get_points(new_box))
        return pixels

    def get_2dbox(self, tolerance=0):
        box_pixels_3d = np.array(self.project_3dbox_into_image())
        max = np.max(box_pixels_3d, axis=0)
        min = np.min(box_pixels_3d, axis=0)
        x = [min[1], min[1], max[1], max[1]]  # , min[1]]
        y = [max[0], min[0], min[0], max[0]]  # , max[0]]
        pixels = np.append(np.asarray(x).reshape(4, 1),
                           np.asarray(y).reshape(4, 1), axis=1)
        return pixels

    def get_semantic_objects(self):
        self.seperator = Separator(data)
        self.seperator.separate_all_labels()

    def plot_data(self):
        plt.imshow(data.image)
        for class_ in self.seperator.semantic_objects:
            class_objects = self.seperator.semantic_objects[class_]
            for object in class_objects:
                try:
                    x = object.bounding_pixel_coords[:, 0]
                    y = object.bounding_pixel_coords[:, 1]
                    plt.plot(x, y, color='r')
                except:
                    continue
        plt.axis('off')
        plt.show()

    def place_crop_into_image(self):
        pass


gen = SyntheticDataGenerator(data)
gen.plot_data()