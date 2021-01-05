from semantic_partitioner import *
import os


"""
Module for undistorting images in the A2D2-Dataset.
"""

path = './A2D2-Dataset/camera_lidar_semantic_bboxes'
directories = os.listdir(path)


# creates image folders
def create_folders(name):
    try:
        for dir in directories:
            subpath = os.path.join(path, dir)
            if name in os.listdir(subpath):
                print(f'name "{name}" is already used. Use another name!')
                raise AssertionError(f'name {name} is already used. Use another name!')
            if os.path.isdir(subpath):
                os.mkdir(os.path.join(subpath, name))
                os.mkdir(os.path.join(subpath, name, 'cam_front_center'))
    except:
        pass


# Directories should be emptied before deletion
def empty_folder(path):
    if os.listdir(path):
        for file in os.listdir(path):
            subpath = os.path.join(path, file)
            if os.path.isdir(subpath):
                empty_folder(subpath)
                os.rmdir(os.path.join(path, file))
            else:
                os.remove(os.path.join(path, file))


# Deletes image folders. It may be needed.
def del_folders(name):
    for dir in directories:
        subpath = os.path.join(path, dir)
        if os.path.isdir(subpath):
            sub_directories = os.listdir(subpath)
            if name in sub_directories:
                rgb_path = os.path.join(subpath, name)
                empty_folder(rgb_path)
                os.rmdir(rgb_path)


# Creates undistorted images.
def generate_undistorted_images(folder_name):
    create_folders(folder_name)
    created = 0
    for dir in directories:
        subpath = os.path.join(path, dir)
        if os.path.isdir(subpath):
            num = len(os.listdir(os.path.join(subpath, 'lidar', 'cam_front_center')))
            for i in range(num):
                data = Loader(subpath, i)
                image = undistort_image(data.image, 'front_center')
                f_name = data.file_names_rgb[i].split('\\')[-1]
                plt.imsave(os.path.join(subpath, folder_name, 'cam_front_center', f_name), image)
                created += 1
                print(f'Current: {f_name}')
                print(f'Created undistorted images: {created}')
                print('--------------------------')


# create_folders('car')
# del_folders('rgb')
# generate_undistorted_images('rgb')