from project_utils import *
import matplotlib.pyplot as plt


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

        # Lidar Data
        self.file_names_lidar = sorted(glob.glob(join(root_path, 'Lidar/*.npz')))
        self.pointcloud = np.load(self.file_names_lidar[id])

        # 3D Bounding Boxes
        self.file_names_3dbox = sorted(glob.glob(join(root_path, '3DLabel/*.json')))
        self.boxes = read_bounding_boxes(self.file_names_3dbox[id])

        # Images
        self.file_names_rgb = sorted(glob.glob(join(root_path, 'RGB/*.png')))
        self.image = plt.imread(self.file_names_rgb[id])

        # Labels
        self.file_names_2dlabel = sorted(glob.glob(join(root_path, 'Label/*.png')))
        self.label = plt.imread(self.file_names_2dlabel[id])

# data = Loader('./A2D2-Dataset/Dataset/', 13)
