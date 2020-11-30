from semantic_partitioner import *
from project_utils import *
import random as rd


class ForegroundObject:
    """
    class for a foreground object with its 3d box, image crop and pc points.
    used in the module synthetic_data_generator.py to be placed in a given scene
    """
    def __init__(self):
        self.box = {}
        self.crop = []
        self.pc_points = []
        self.old_pos = []

    # 3D bounding box
    def set_box(self, box):
        self.box = box

    # Image crop
    def set_crop(self, crop):
        self.crop = crop

    # PC points
    def set_pc_points(self, points):
        self.pc_points = points

    # Old position in pc
    def set_old_position(self, old_pos):
        self.old_pos = old_pos

class ObjectsGenerator:
    """
    class for extracting foreground objects out of a given scene. found objects
    are listed in the variable self.forground_objects.
    Criteria for including a given object:
    - No occlusion
    - No truncation
    """
    def __init__(self, data):
        self.foreground_objects = []
        self.data = data
        self.find_foreground_objects(data)

        print(f"{len(self.foreground_objects)} foreground objects were found.")

    def get_boxes(self, data):
        for box in data.boxes:
            if not box['occlusion'] and not box['truncation']:
                foreground_obj = ForegroundObject()
                foreground_obj.set_box(box)
                foreground_obj.set_old_position(box['center'].copy())
                self.foreground_objects.append(foreground_obj)

    # Crops the image of an object in the scene given its 2d box
    def get_crops(self, data):
        seperator = Separator(data)
        for fg_obj in self.foreground_objects:
            color = self.get_color(fg_obj)
            x, y, gray = seperator.find_class_pixels(color)
            gray = self.turn_to_4channel(gray)
            background = data.image
            # background = undistort_image(background, 'front_center')
            background_1 = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA).copy()
            image = np.where(gray == [1, 1, 1, 1], background_1, [0, 230 / 255, 64 / 255, 0])
            box = fg_obj.box
            x1, x2 = int(box['left']), int(box['right'])
            y1, y2 = int(box['top']), int(box['bottom'])
            image = image[x1:x2, y1:y2, :]
            fg_obj.set_crop(image)

    # Gets pc points of the box in the original scene
    def get_pc_points(self):
        for fg_obj in self.foreground_objects:
            box = fg_obj.box
            points = self.data.pointcloud['points']
            pts = get_points(box)
            lower = np.min(pts, axis=0)
            upper = np.max(pts, axis=0)
            s = np.logical_and(points > lower, points < upper)
            s = np.all(s, axis=1)
            l_pc = points[s]
            obj = l_pc.copy()
            fg_obj.set_pc_points(obj)

    # Finds out the class color of a foreground object from
    # its semantic segmentation
    def get_color(self, fg_obj):
        box = fg_obj.box
        x = int((box['right'] + box['left'])/2)
        y = int((box['bottom'] + box['top'])/2)
        rgb = self.data.label[x][y]
        color = self.rgb_to_hex(rgb)
        return color

    # Converts an rgb to hexadecimal color. Used for getting the right
    # subclass out of the semantic segmentation
    def rgb_to_hex(self, rgb):
        rgb = (rgb * 255).astype(np.uint8)
        rgb = (rgb[0], rgb[1], rgb[2])
        return '#%02x%02x%02x' % rgb

    # Turns an image from 3 to 4 channels
    def turn_to_4channel(self, img):
        arr = img
        arr = arr.reshape(img.shape[0], img.shape[1], 1)
        arr_2 = np.append(arr, arr, axis=2)
        arr_2 = np.append(arr_2, arr, axis=2)
        ones = np.ones((img.shape[0], img.shape[1]))
        arr_2 = np.append(arr_2, ones.reshape(ones.shape[0], ones.shape[1], 1), axis=2)
        return arr_2

    # Finds objects in the scene
    def find_foreground_objects(self, data):
        self.get_boxes(data)
        self.get_crops(data)
        self.get_pc_points()

    # Plots foreground objects in the scene
    def plot_foreground_object(self, id):
        try:
            plt.imshow(self.foreground_objects[id].crop)
            print(self.foreground_objects[id].box['class'])
            plt.show()
        except:
            pass

    def save_foregrounds(self):
        frame_id = self.data.file_names_lidar[self.data.id].split('\\')[-1]
        frame_id = str(frame_id)[:-4]
        for i in range(len(self.foreground_objects)):
            pickle.dump(self.foreground_objects[i], open(f'./foregrounds/{frame_id}_{i}.fg', "wb"))
            plt.imsave(f'./foregrounds/{frame_id}_{i}.png', self.foreground_objects[i].crop)



# Plot results
# data = Loader('./A2D2-Dataset/Dataset/', 1)
# foreground_objs = ObjectsGenerator(data)
# # foreground_objs.plot_foreground_object(0)
# print(foreground_objs.foreground_objects[0].pc_points.shape)
# print(data.pointcloud['points'].shape)
# print(np.append(data.pointcloud['points'],foreground_objs.foreground_objects[0].pc_points, axis=0 ).shape)
# # plot_point_cloud(foreground_objs.foreground_objects[2].pc_points)

#-----------------------
#   save foregrounds
#-----------------------

data_2 = Loader(path, 0)
scenes = data_2.folder_names_scene
path = './A2D2-Dataset/A2D2/camera_lidar_semantic_bboxes/camera_lidar_semantic_bboxes/'

for i in scenes:
    subpath = path + i
    data_1 = Loader(subpath, 0)
    range_ = len(data_1.file_names_3dbox)
    for i in range(range_):
        try:
            data = Loader(subpath, i)
            foreground_objs = ObjectsGenerator(data)
            foreground_objs.save_foregrounds()
        except:
            continue