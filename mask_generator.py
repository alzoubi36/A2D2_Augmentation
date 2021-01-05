from semantic_partitioner import *
import os

# Adjust desired classes that should appear in the masks
car = True
pedestrian = True
bicycle = False

# Adjust desired color for the objects: black or whíte
black = False

path = './A2D2-Dataset/camera_lidar_semantic_bboxes'
directories = os.listdir(path)


# creates mask folders
def create_mask_folders():
    try:
        for dir in directories:
            subpath = os.path.join(path, dir)
            if os.path.isdir(subpath):
                os.mkdir(os.path.join(subpath, 'mask'))
    except:
        pass


# deletes mask folders. It may be needed.
def del_mask_folders():
    for dir in directories:
        subpath = os.path.join(path, dir)
        if os.path.isdir(subpath):
            sub_directories = os.listdir(subpath)
            if 'mask' in sub_directories:
                masks_path = os.path.join(subpath, 'mask')
                for file in os.listdir(masks_path):
                    os.remove(os.path.join(masks_path, file))
                os.rmdir(os.path.join(subpath, 'mask'))


# Creates class specific masks.
# Available classes: car, pedestrian, bicycle
def create_specific_masks(car=car, pedestrian=pedestrian, bicycle=bicycle, black=black):
    create_mask_folders()
    created = 0
    for dir in directories:
        subpath = os.path.join(path, dir)
        if os.path.isdir(subpath):
            num = len(os.listdir(os.path.join(subpath, 'lidar', 'cam_front_center')))
            for i in range(num):
                data = Loader(subpath, i)
                sep = Separator(data)
                mask = sep.create_specified_mask()
                mask = mask.astype('uint8')*255
                if black:
                    mask = np.where(mask == 255, 0, 255).astype('uint8')
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                f_name = data.file_names_rgb[i].split('\\')[-1]
                plt.imsave(os.path.join(subpath, 'mask', f_name), mask)
                created += 1
                print(f'Current: {f_name}')
                print(f'Created masks: {created}')
                print('--------------------------')


# create_mask_folders()
del_mask_folders()
# create_specific_masks()
