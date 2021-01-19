from semantic_partitioner import *
from project_utils import *
from pc_to_image_projector import *
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
        self.rel_pos_2d = []

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

    # Relative position to lidar 2dbox (projected 3dbox)
    def set_rel_pos_to_projected_3dbox(self, pos):
        self.rel_pos_2d = pos



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
        """
        Finds allowed 3d-boxes and creates associated ForegroundObject objects
        :param data: as loaded with Loader
        :return: None
        """
        for box in data.boxes:
            if not box['occlusion'] and not box['truncation']:
                foreground_obj = ForegroundObject()
                foreground_obj.set_box(box)
                foreground_obj.set_old_position(box['center'].copy())
                self.foreground_objects.append(foreground_obj)

    def get_crops(self, data):
        """
        Crops the image of an object in the scene given its 2d box
        :param data: as loaded with Loader
        :return: None
        """
        seperator = Separator(data)
        for fg_obj in self.foreground_objects:
            color = self.get_color(fg_obj)
            x, y, mask = seperator.find_class_pixels(color)
            background = data.image * 255
            background_1 = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA).copy()
            condition = np.where(mask == 1, 255, 0).astype('uint')
            image = background_1.astype('uint8')
            image[:, :, 3] = condition
            box = fg_obj.box
            box_pixels_3d = project_box_from_pc_to_image(get_points(box), undist=True)
            max_ = np.max(box_pixels_3d, axis=0)
            min_ = np.min(box_pixels_3d, axis=0)
            x1, x2 = int(min_[1]), int(max_[1])
            y1, y2 = int(min_[0]), int(max_[0])
            center_lidar_2d_box = np.array(((x1 + x2) / 2, (y1 + y2) / 2))
            contours = seperator.find_contours(mask)
            rects = []
            dists = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                center_sem_2d_box = np.array((y + h / 2, x + w / 2))
                rects.append((x, y, w, h, center_sem_2d_box))
                dists.append(cv2.norm(center_lidar_2d_box -
                                      center_sem_2d_box))

            xs_1, ys_1, w, h, center_sem_2d_box = rects[dists.index(min(dists))]

            # rel_pos_to_projected_3dbox = np.asarray(center_lidar_2d_box - center_sem_2d_box)
            rel_pos_to_projected_3dbox = np.asarray([y1 - xs_1, x1 - ys_1])
            print('Sem: ', ys_1, ys_1+h, xs_1, xs_1+w)
            print('Lider: ', x1, x2, y1, y2)
            print('rel_pos: ', rel_pos_to_projected_3dbox)
            print('center sem: ', center_sem_2d_box)
            print('center lidar: ', center_lidar_2d_box)
            plt.imshow(data.image)
            plt.scatter(center_sem_2d_box[1], center_sem_2d_box[0])
            plt.scatter(center_lidar_2d_box[1], center_lidar_2d_box[0])
            # plt.show()
            image = image[ys_1:ys_1+h, xs_1:xs_1+w, :]
            fg_obj.set_crop(image)
            fg_obj.set_rel_pos_to_projected_3dbox(rel_pos_to_projected_3dbox)

    def get_pc_points(self):
        """
        Gets local pc of 3d-boxes in the scene
        :return: None
        """
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

    def get_color(self, fg_obj):
        """
        Finds out the class color of a foreground object from
        its semantic segmentation
        :param fg_obj: Foreground object <ForegroundObject>
        :return: color in hexadecimal form
        """
        box = fg_obj.box
        x = int((box['right'] + box['left']) / 2)
        y = int((box['bottom'] + box['top']) / 2)
        rgb = self.data.label[x][y]
        color = self.rgb_to_hex(rgb)
        return color

    def rgb_to_hex(self, rgb):
        """
        Converts an rgb to hexadecimal color. Used for getting the right
        subclass out of the semantic segmentation
        :param rgb: color [r, g, b]
        :return: hexadecimal color
        """
        rgb = (rgb * 255).astype(np.uint8)
        rgb = (rgb[0], rgb[1], rgb[2])
        return '#%02x%02x%02x' % rgb

    def turn_to_4channel(self, img):
        """
        Turns an image from 3 to 4 channels
        :param img: image with 3 channels <ndarray>
        :return: 4 channel image <ndarray>
        """
        arr = img
        arr = arr.reshape(img.shape[0], img.shape[1], 1)
        arr_2 = np.append(arr, arr, axis=2)
        arr_2 = np.append(arr_2, arr, axis=2)
        ones = np.ones((img.shape[0], img.shape[1]))
        arr_2 = np.append(arr_2, ones.reshape(ones.shape[0], ones.shape[1], 1), axis=2)
        return arr_2

    def find_foreground_objects(self, data):
        """
        Finds foreground objects in the scene
        :param data: as loaded with Loader
        :return: None
        """
        self.get_boxes(data)
        self.get_crops(data)
        self.get_pc_points()

    def check_color(self, overlay, to_find, label=False):
        """
        finds a color in an image.
        :param overlay: <image>
        :param to_find: <RGBA Color list like>
        :return: logical array with the same dimension as overlay
        """
        arr = np.all(overlay == to_find, axis=2)
        arr = arr.reshape(overlay.shape[0], overlay.shape[1], 1)
        arr = np.append(arr, arr, axis=2)
        arr = np.append(arr, arr, axis=2)
        return arr

    def plot_foreground_object(self, id):
        """
        Plots foreground objects in the scene
        :param id: id of the desired foreground object
        :return: None
        """
        try:
            plt.imshow(self.foreground_objects[id].crop)
            print(self.foreground_objects[id].box['class'])
            plt.show()
        except:
            pass

    def save_foregrounds(self):
        """
        saves found foreground objects in the scene. included files:
        - PNG-image of cropped object
        - FG-File contains an object of the ForegroundObject class
        :return: None
        """
        frame_id = self.data.file_names_lidar[self.data.id].split('\\')[-1]
        frame_id = str(frame_id)[:-4]
        try:
            for i in range(len(self.foreground_objects)):
                pickle.dump(self.foreground_objects[i], open(f'./foregrounds/{frame_id}_{i}.fg', "wb"))
                plt.imsave(f'./foregrounds/{frame_id}_{i}.png', self.foreground_objects[i].crop)
        except:
            import os
            os.mkdir('foregrounds')
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

# -----------------------
#   save foregrounds
# -----------------------

def create_foregrounds(path):
    """
    iterates through the whole dataset and saves foreground objects
    :param path: to the dataset
    :return: None
    """
    data_2 = Loader(path, 0)
    scenes = data_2.folder_names_scene
    path = './A2D2-Dataset/camera_lidar_semantic_bboxes/'
    for i in scenes:
        subpath = path + i
        data_1 = Loader(subpath, 0)
        range_ = len(data_1.file_names_3dbox)
        for i in range(range_):
            # try:
            data = Loader(subpath, i)
            # plt.imshow(data.image)
            # plt.show()
            foreground_objs = ObjectsGenerator(data)
            # print(f'scene {i}')
            foreground_objs.save_foregrounds()
            # except:
            #     continue

# create_foregrounds(path)
# data = Loader(path, 0)
# foreground_objs = ObjectsGenerator(data)1
# foreground_objs.save_foregrounds()
