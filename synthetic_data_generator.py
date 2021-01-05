from _3d_bbox_position_finder import *
from semantic_partitioner import *
from objects_generator import *
import pickle
import os

data = Loader(path, 2)
# Class for generating synthetic data for the A2D2-Dataset.
class SyntheticDataGenerator:
    """
    Class for generating synthetic data for the A2D2-Dataset. Uses other Modules of the project.
    """
    def __init__(self, data):
        self.data = data
        self.points = data.pointcloud['points']
        self.get_semantic_objects()

        # self.load_foreground('20180810142822_lidar_frontcenter_000027828_0')
        self.load_foreground('20180807145028_lidar_frontcenter_000002019_0')
        self.load_foreground('20180925112730_lidar_frontcenter_000018730_0')
        # self.load_foreground('20180807145028_lidar_frontcenter_000002031_0')
        # self.load_foreground('20180807145028_lidar_frontcenter_000030332_0') # Fehler !!! beim Spiegeln
        # self.load_foreground('20180925112730_lidar_frontcenter_000032670_0') # Fehler !!! beim Spiegeln
        # self.load_foreground('20180807145028_lidar_frontcenter_000019140_2')

    def propose_3dbbox_position(self):
        """
        Proposes a new 3D box position
        :return: position in pc in format [x, y]
        """
        return find_position(self.points, self.box, self.old_pos, self.old_angle)

    def project_3dbox_into_image(self):
        """
        Projects a given 3d box into image
        :return: 3d-box pixels <ndarray>
        """
        new_position = self.propose_3dbbox_position()
        # new_position = (self.old_pos[0], self.old_pos[1])
        new_box = self.box.copy()
        print('old angle: ', np.degrees(self.old_angle))
        points2 = get_points(self.box)
        id = 2
        angle = new_position_rotation_angle(self.box, new_position, self.old_pos)
        self.box = update_rotation(self.box, angle, self.old_angle)
        self.box['center'][0] = new_position[0]
        self.box['center'][1] = new_position[1]
        points = get_points(self.box)
        print('-------------------')
        print('distance old:', np.linalg.norm(points2[id]))
        print('distance new:', np.linalg.norm(points[id]))
        print('-------------------')
        print('ref. point old:', points2[id])
        print('ref. point new:', points[id])
        # print('points_old: ', points2)
        # print('points_new: ', points)
        # print('-------------------')
        pixels = project_box_from_pc_to_image(points, undist=True)
        return pixels

    # Creates a 2D-box out of a 3D box
    def get_2dbox(self, tolerance=0):
        box_pixels_3d= np.array(self.project_3dbox_into_image())
        max = np.max(box_pixels_3d, axis=0) + tolerance
        min = np.min(box_pixels_3d, axis=0) - tolerance
        x = [min[1], min[1], max[1], max[1]]  # , min[1]]
        y = [max[0], min[0], min[0], max[0]]  # , max[0]]
        pixels = np.append(np.asarray(x).reshape(4, 1),
                           np.asarray(y).reshape(4, 1), axis=1)
        return pixels, box_pixels_3d

    # Returns depth of a given 3D Box
    # Required to get occluded pixels of the box
    def get_box_depth(self, points):
        sep = Separator(data)
        depths = np.array([sep.calculate_depth_value(point) for point in points])
        depth = np.mean(depths)
        return depth

    # Hides image pixels that intersect with the semantic object
    def hide_pixels(self, sem_object, augmented, pos):
        indices = sem_object.pixels
        width = augmented.shape[1]
        height = augmented.shape[0]
        # print('shape: ', augmented.shape)
        # print('width: ', width)
        # print('height: ', height)
        for i in indices:
            x = int(i[0] - pos[0])
            y = int(i[1] - pos[1])
            if x > 0 and y > 0:
                if x < width and y < height:
                    # print('x: ', x)
                    # print('y: ', y)
                    augmented[y][x] = [0, 0, 0, 0]
        return augmented

    def hide_all_pixels(self, overlay, depth, pos):
        allowed = ['Car 1', 'Car 2', 'Car 3', 'Car 4']
        for class_ in class_list.values():
            if class_ in allowed:
                for sem_obj in self.semantic_objects[class_]:
                    try:
                        occluded = self.occludes_box(sem_obj, depth)
                        if occluded:
                            overlay = self.hide_pixels(sem_obj, overlay, pos)
                    except:
                        continue
        return overlay

    # Checks if the semantic object occludes box
    def occludes_box(self, sem_object, box_depth):
        if box_depth > sem_object.depth:
            return True
        else:
            return False

    # Tries to find saved semantic objects on the hard disk
    # if nothing is found it calculates them
    def get_semantic_objects(self):
        f_name = data.file_names_rgb[data.id].split('\\')[-1]
        f_name = './background_objs/' + f_name[:-4] + '_semantic_objects.sp'

        try:
            self.semantic_objects = pickle.load(open(f_name, "rb"))
        except:
            self.create_folder('background_objs')
            self.create_semantic_objs(f_name)

    def create_semantic_objs(self, f_name):
        self.seperator = Separator(data)
        self.seperator.separate_all_labels()
        self.semantic_objects = self.seperator.semantic_objects
        pickle.dump(self.seperator.semantic_objects, open(f_name, "wb"))

    # Plots all semantic object bounding pixels in the image
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

    # Plots 2D and 3D boxes
    def plot_2d_3d_boxes(self, pixels_2d, pixels_3d, _2d = False):
        plt.plot(pixels_3d[:, 0], pixels_3d[:, 1], linewidth=0.3, color="r")
        if _2d:
            plt.plot(pixels_2d[:, 1], pixels_2d[:, 0], color="b", linewidth=0.8)
            plt.plot([pixels_2d[-1][1], pixels_2d[0][1]],
                     [pixels_2d[-1][0], pixels_2d[0][0]], color="b", linewidth=0.8)
        # pts_3d = np.array([[20, 0, 0], [100, 0, 100]])
        # pts_2d = np.array([project_from_pc_to_image(cam_matrix_dist, i) for i in pts_3d])
        # print(pts_2d)
        # plt.plot(pts_2d[:, 1], pts_2d[:, 0])
        # plt.scatter(pixels_2d[0][0], pixels_2d[0][1])
        # plt.scatter(pixels_2d[1][0], pixels_2d[1][1], color='r')
        # plt.scatter(pixels_2d[2][0], pixels_2d[2][1], color='r')
        # plt.scatter(pixels_2d[0][0], pixels_2d[0][1], color='r')
        # plt.imshow(self.data.image)
        plt.axis('off')
        # plt.show()

    # Overlays a given foreground on a background
    def overlay_foreground_background(self, img, img_overlay, pos):
        """Overlay img_overlay on top of img at the position specified by
            pos and blend using alpha_mask.

            Alpha mask must contain values within the range [0, 1] and be the
            same size as img_overlay.
            """

        x, y = int(pos[0]), int(pos[1])

        alpha_mask = img_overlay[:, :, 3]

        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            print('None!')
            return

        channels = img.shape[2]

        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha

        for c in range(channels):
            img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                    alpha_inv * img[y1:y2, x1:x2, c])
        return img

    def adjust_foreground(self, overlay, to_del, label=False):
        """
         Adjusts the alpha channel of a 4 channel image. Hides background color.
         Hides occluded pixels of the newly placed object
        :param overlay: <image>
        :param to_del: <RGBA Color list like>
        :return: adjusted foreground image
        """
        if label:
            arr = overlay[:, :, 3]
            arr = arr.reshape(overlay.shape[0], overlay.shape[1], 1)
            arr = np.append(arr, arr, axis=2)
            arr = np.append(arr, arr, axis=2)
            overlay = np.where(arr, to_del, [0, 0, 0, 0])
        else:
            arr = np.all(overlay == to_del, axis=2)
            arr = arr.reshape(overlay.shape[0], overlay.shape[1], 1)
            arr = np.append(arr, arr, axis=2)
            arr = np.append(arr, arr, axis=2)
            overlay = np.where(arr, [0, 0, 0, 0], overlay)
        return overlay

    def rescale_foreground(self, foreground, scale_percent):
        """
        rescales a the given foreground to suit the new distance from sensor.
        :param foreground: <image>
        :param scale_percent: <float>
        :return: rescaled foreground <image>
        """
        width = int(foreground.shape[1] * scale_percent)
        height = int(foreground.shape[0] * scale_percent)
        dim = (width, height)
        resized = cv2.resize(foreground, dim, interpolation=cv2.INTER_AREA)
        return resized

    # Calculates new size of the image crop in the new postition
    def get_new_scale(self, box_pixels_3d):
        max_1 = np.max(box_pixels_3d, axis=0)
        min_1 = np.min(box_pixels_3d, axis=0)

        # Show thing
        # plt.imshow(data.image)
        # plt.scatter(max_1[0], max_1[1])
        # plt.scatter(min_1[0], min_1[1], color='r')
        # plt.show()

        h_new = max_1[1] - min_1[1]
        w_new = max_1[0] - min_1[0]
        # print('------------------')
        # print('new height: ', h_new)
        box_pixels = project_box_from_pc_to_image(get_points(self.old_box), undist=True)
        max_0 = np.max(box_pixels, axis=0)
        min_0 = np.min(box_pixels, axis=0)

        # plt.imshow(data.image)
        # plt.scatter(max_0[0], max_0[1])
        # plt.scatter(min_0[0], min_0[1], color='r')
        # plt.show()


        h_old = max_0[1] - min_0[1]
        w_old = max_0[0] - min_0[0]
        # print('old height: ', h_old)
        # print('------------------')
        # print('new width: ', w_new)
        # print('old width: ', w_old)
        # print('------------------')
        # print('old ratio: ', w_old/h_old)
        # print('new ratio: ', w_new/h_new)
        # print('------------------')
        scale = w_new / w_old
        return abs(scale)

    # Generate a synthetic image by putting the new object in the scene.
    def generate_synthetic_scene(self, save=False):
        print('creating synthetic scene ...')
        pixels = [1, [-3000, 0]]

        # Insure that the proposed box can be seen in image
        while pixels[1][0] < -200 or pixels[1][0] > 1900:
            pixels, box_pixels_3d= self.get_2dbox()

        scale = self.get_new_scale(box_pixels_3d)
        overlay = self.foreground
        overlay = self.adjust_alpha(overlay, 1, full=False)
        synthetic_img = self.get_rgb(overlay, scale, pixels)
        synthetic_label = self.get_label(overlay, scale, pixels)
        self.plot_2d_3d_boxes(pixels, box_pixels_3d, _2d=True)
        if save:
            self.save_synthetic_scene(synthetic_img, synthetic_label, self.box)
        return synthetic_img, synthetic_label

    # Load a foreground object
    def load_foreground(self, name):
        self.foreground = cv2.imread(f'./foregrounds/{name}.png', cv2.IMREAD_UNCHANGED)
        foreground_obj = pickle.load(open(f'./foregrounds/{name}.fg', "rb"))
        foreground_obj2 = pickle.load(open(f'./foregrounds/{name}.fg', "rb"))
        self.box, self.foreground_pc = foreground_obj.box, foreground_obj.pc_points
        self.old_box = foreground_obj2.box.copy()
        self.old_pos = self.old_box['center']
        self.old_angle = self.old_box['angle']

    # Todo: to be written efficiently : get_rgb & get_label
    def get_rgb(self, overlay, scale, pixels):
        background = self.data.image
        background_1 = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA).copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGRA).copy()
        # overlay = self.adjust_foreground(overlay, [0, 230, 64, 255]) / 255
        overlay = overlay/255
        print('scale:', scale)
        print('old shape:', overlay.shape)
        overlay = self.rescale_foreground(overlay, scale)
        print('new shape:', overlay.shape)
        depth = self.get_box_depth(get_points(self.box))
        # x = int((pixels[2][0] + pixels[1][0])/2 - overlay.shape[1]/2)
        # y = int((pixels[0][1] + pixels[1][1])/2 - overlay.shape[0]/2)
        # pos = (x, y)
        pos = round(pixels[1][1]), round(pixels[1][0])
        overlay = self.hide_all_pixels(overlay, depth, pos)
        synthetic_img = self.overlay_foreground_background(background_1, overlay, pos)
        synthetic_img[:, :, 3] = np.where(synthetic_img[:, :, 3] > 1, .99, synthetic_img[:, :, 3])
        synthetic_img = np.where(synthetic_img < 0, 0, synthetic_img)
        return synthetic_img

    def get_label(self, overlay, scale, pixels):
        background = self.data.label
        background_1 = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA).copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGRA).copy()
        overlay = self.adjust_foreground(overlay, [0, 230, 64, 255], label=True) / 255
        overlay = self.rescale_foreground(overlay, scale)
        depth = self.get_box_depth(get_points(self.box))
        pos = int(pixels[1][0]), int(pixels[1][1])
        overlay = self.hide_all_pixels(overlay, depth, pos)
        synthetic_label = self.overlay_foreground_background(background_1, overlay, pos)
        try:
            synthetic_label[:, :, 3] = np.where(synthetic_label[:, :, 3] > 1, .99, synthetic_label[:, :, 3])
            synthetic_label = np.where(synthetic_label < 0, 0, synthetic_label)
        except:
            pass
        return synthetic_label

    def get_pc(self, box, points):
        points[:, 0] = points[:, 0] + box['center'][0]
        points[:, 1] = points[:, 1] + box['center'][1]
        points = np.dot(points, box['rotation'].T)
        return points

    def save_synthetic_scene(self, synthetic_img, synthetic_label, box):
        i = rd.random()
        i = str(i)[-4:]
        self.create_folder('synthetic data')
        sub_folders = ['camera', 'label', 'label3D', 'lidar']
        try:
            for f in sub_folders:
                self.create_folder('synthetic data/' + f)
        except:
            pass
        plt.imsave(f'./synthetic data/camera/img_{i}.png', synthetic_img)
        plt.imsave(f'./synthetic data/Label/label_{i}.png', synthetic_label)
        boxes = data.boxes.append(box)
        json.dump(boxes, open(f"./synthetic data/label3D/3dbox_{i}.json", "w"))
        foreground_pc = self.get_pc(box, self.foreground_pc)
        pointcloud = dict(self.data.pointcloud)
        pointcloud['points'] = np.append(self.data.pointcloud['points'],
                                         foreground_pc, axis=0)
        np.savez_compressed(f"./synthetic data/lidar/lidar_{i}", pointcloud)

    @staticmethod
    def create_folder(name):
        if name not in os.listdir('./'):
            os.mkdir(name)

    def adjust_alpha(self, img, a, full=False):
        if full:
            img[:, :, 3] = img[:, :, 3] * 0
            img[:, :, 3] = img[:, :, 3] + 255
        else:
            img[:, :, 3] = img[:, :, 3] * a
        return img



gen = SyntheticDataGenerator(data)
# for i in range(20):
#     try:
#         synthetic_img = gen.generate_synthetic_image()
#         print('max: ', synthetic_img.max())
#         print('min: ', synthetic_img.min())
#     except:
#         continue
synthetic_img, synthetic_label = gen.generate_synthetic_scene(save=False)
# plt.imsave(f'./synthetic data/RGB/synthetic_img.png', synthetic_img)
# plt.imsave(f'./synthetic data/Label/synthetic_label.png', synthetic_label)

plt.imshow(synthetic_img)
# plt.imsave(f'./synthetic data/synthetic_img.png', data.label)
plt.axis('off')
plt.show()

# plt.imshow(synthetic_label)
# plt.savefig(f'./Test_beispiele/fig_{rd.randint(0,100)}.png', dpi=300)
# plt.show()
