from project_utils import *
import matplotlib.pyplot as plt
import os

# Class to load data from a defined directory
class Loader:
    def __init__(self, path, id):
        self.path = path
        self.id = id  # Frame ID
        self.load_data()

    # Loads all available data belonging to a Frame
    def load_data(self):
        root_path = self.path
        id = self.id
        self.folder_names_scene = os.listdir('./A2D2-Dataset/camera_lidar_semantic_bboxes/')[:-2]
        # print(self.folder_names_scene)

        # Lidar Data
        self.file_names_lidar = sorted(glob.glob(join(root_path, 'lidar/cam_front_center/*.npz')))
        self.pointcloud = np.load(self.file_names_lidar[id])

        # 3D Bounding Boxes
        self.file_names_3dbox = sorted(glob.glob(join(root_path, 'label3D/cam_front_center/*.json')))
        self.boxes = read_bounding_boxes(self.file_names_3dbox[id])

        # Images
        self.file_names_rgb = sorted(glob.glob(join(root_path, 'camera/cam_front_center/*.png')))
        self.image = plt.imread(self.file_names_rgb[id])

        # Labels
        self.file_names_2dlabel = sorted(glob.glob(join(root_path, 'label/cam_front_center/*.png')))
        self.label = plt.imread(self.file_names_2dlabel[id])


path = './A2D2-Dataset/camera_lidar_semantic_bboxes/20180807_145028/'
# data = Loader(path, 13)
