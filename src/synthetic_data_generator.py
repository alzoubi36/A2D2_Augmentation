from _3d_bbox_position_finder import *
from semantic_partitioner import *
from objects_generator import *
import pickle

print(path)
data = Loader(path, 10)

# Class for generating synthetic data for the A2D2-Dataset.
class SyntheticDataGenerator:
    def __init__(self, data):
        self.data = data
        self.pointcloud = data.pointcloud
        self.points = self.pointcloud['points']
        self.label = data.label
        self.get_semantic_objects()
        self.load_foreground('4268_0')

        # Todo: create objects for augmentation (boxes, labels and crops)
        # self.box = Loader('./A2D2-Dataset/Dataset/', 11).boxes[5]

    # Proposes a new 3D box position
    def propose_3dbbox_position(self):
        return find_position(self.points, box, self.old_pos)

    # Projects a given 3d box into image
    def project_3dbox_into_image(self):
        new_position = self.propose_3dbbox_position()
        # new_box = self.box.copy()
        self.box['center'][0] = new_position[0]
        self.box['center'][1] = new_position[1]
        angle = new_position_rotation_angle(self.box, new_position, self.old_pos)
        update_rotation(self.box, angle)
        points = get_points(self.box)
        pixels = project_box_from_pc_to_image(points, cam_matrix_dist)
        return pixels, points

    # Creates a 2D-box out of a 3D box
    def get_2dbox(self, tolerance=0):
        box_pixels_3d, points = np.array(self.project_3dbox_into_image())
        max = np.max(box_pixels_3d, axis=0) + tolerance
        min = np.min(box_pixels_3d, axis=0) - tolerance
        x = [min[1], min[1], max[1], max[1]]  # , min[1]]
        y = [max[0], min[0], min[0], max[0]]  # , max[0]]
        pixels = np.append(np.asarray(x).reshape(4, 1),
                           np.asarray(y).reshape(4, 1), axis=1)
        return pixels, box_pixels_3d, points

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
        print('shape: ', augmented.shape)
        print('width: ', width)
        print('height: ', height)
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
        f_name = data.file_names_3dbox[data.id][:-4]
        try:
            self.semantic_objects = pickle.load(open(f_name, "rb"))
        except:
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
    def plot_2d_3d_boxes(self, pixels_2d, pixels_3d):
        plt.plot(pixels_3d[:, 1], pixels_3d[:, 0], linewidth=0.5)
        plt.plot(pixels_2d[:, 0], pixels_2d[:, 1], color="r", linewidth=1)
        plt.plot([pixels_2d[-1][0], pixels_2d[0][0]],
                 [pixels_2d[-1][1], pixels_2d[0][1]], color="r", linewidth=1)
        # plt.scatter(pixels_2d[0][0], pixels_2d[0][1])
        # plt.scatter(pixels_2d[1][0], pixels_2d[1][1], color='r')
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
        arr = np.all(overlay == to_del, axis=2)
        arr = arr.reshape(overlay.shape[0], overlay.shape[1], 1)
        arr = np.append(arr, arr, axis=2)
        arr = np.append(arr, arr, axis=2)
        if label:
            overlay = np.where(arr, [0, 0, 0, 0], to_del)
        else:
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
    def get_new_scale(self, y1, y2):
        h_new = y2 - y1
        h_old = self.box['right'] - self.box['left']
        scale = h_new / h_old
        return abs(scale)

    # Generate a synthetic image by putting the new object in the scene.
    def generate_synthetic_scene(self, save=False):
        pixels = [1, [-3000, 0]]

        # Insure that the proposed box can be seen in image
        while pixels[1][0] < -200 or pixels[1][0] > 1900:
            pixels, box_pixels_3d, points = self.get_2dbox()

        y2, y1 = pixels[1][1], pixels[0][1]
        scale = self.get_new_scale(y1, y2)
        overlay = self.foreground
        synthetic_img = self.get_rgb(overlay, scale, pixels)
        synthetic_label = self.get_label(overlay, scale, pixels)
        self.plot_2d_3d_boxes(pixels, box_pixels_3d)
        if save:
            self.save_synthetic_scene(synthetic_img, synthetic_label, self.box)
        return synthetic_img, synthetic_label

    # Load a foreground object
    def load_foreground(self, name):
        self.foreground = cv2.imread(f'./foregrounds/{name}.png', cv2.IMREAD_UNCHANGED)
        foreground_obj = pickle.load(open(f'./foregrounds/{name}.fg', "rb"))
        self.box, self.foreground_pc = foreground_obj.box, foreground_obj.pc_points
        self.old_pos = foreground_obj.old_pos

    # Todo: to be written efficiently : get_rgb & get_label
    def get_rgb(self, overlay, scale, pixels):
        background = self.data.image
        print(scale)
        background_1 = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA).copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGRA).copy()
        overlay = self.adjust_foreground(overlay, [0, 230, 64, 255]) / 255
        overlay = self.rescale_foreground(overlay, scale)
        depth = self.get_box_depth(get_points(self.box))
        # pos = int(box_pixels_3d[5][1]), int(box_pixels_3d[5][0])
        pos = int(pixels[1][0]), int(pixels[1][1])
        overlay = self.hide_all_pixels(overlay, depth, pos)
        synthetic_img = self.overlay_foreground_background(background_1, overlay, pos)
        synthetic_img[:, :, 3] = np.where(synthetic_img[:, :, 3] > 1, .99, synthetic_img[:, :, 3])
        synthetic_img = np.where(synthetic_img < 0, 0, synthetic_img)
        return synthetic_img

    def get_label(self, overlay, scale, pixels):
        background = self.data.label
        print(scale)
        background_1 = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA).copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGRA).copy()
        overlay = self.adjust_foreground(overlay, [0, 230, 64, 255], label=True) / 255
        overlay = self.rescale_foreground(overlay, scale)
        depth = self.get_box_depth(get_points(self.box))
        pos = int(pixels[1][0]), int(pixels[1][1])
        overlay = self.hide_all_pixels(overlay, depth, pos)
        synthetic_label = self.overlay_foreground_background(background_1, overlay, pos)
        synthetic_label[:, :, 3] = np.where(synthetic_label[:, :, 3] > 1, .99, synthetic_label[:, :, 3])
        synthetic_label = np.where(synthetic_label < 0, 0, synthetic_label)
        return synthetic_label

    def get_pc(self, box, points):
        points[:, 0] = points[:, 0] + box['center'][0]
        points[:, 1] = points[:, 1] + box['center'][1]
        points = np.dot(points, box['rotation'].T)
        return points

    def save_synthetic_scene(self, synthetic_img, synthetic_label, box):
        i = rd.random()
        i = str(i)[-4:]
        plt.imsave(f'./synthetic data/RGB/img_{i}.png', synthetic_img)
        plt.imsave(f'./synthetic data/Label/label_{i}.png', synthetic_label)
        boxes = data.boxes.append(box)
        json.dump(boxes, open(f"./synthetic data/3DLabel/3dbox_{i}.json", "w"))
        foreground_pc = self.get_pc(box, self.foreground_pc)
        pointcloud = dict(self.data.pointcloud)
        pointcloud['points'] = np.append(self.data.pointcloud['points'],
                                                   foreground_pc, axis=0)
        np.savez_compressed(f"./synthetic data/Lidar/lidar_{i}", pointcloud)


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
plt.show()

plt.imshow(synthetic_label)
# plt.show()
