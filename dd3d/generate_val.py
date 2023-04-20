import os
import shutil

root = '/home/carla/admt_student/team3_ss23/data/KITTI3D'
os.makedirs(os.path.join(root, "val"))
os.makedirs(os.path.join(root, "val", "calib"))
os.makedirs(os.path.join(root, "val", "image_2"))
os.makedirs(os.path.join(root, "val", "label_2"))

with open(os.path.join(root, "mv3d_kitti_splits", "val.txt")) as _f:
    lines = _f.readlines()
    split = [line.rstrip("\n") for line in lines]

    for sub in ['calib', 'image_2', 'label_2']:
        for file in split:
            if sub == 'calib' or sub == 'label_2':
                file += '.txt'
            else:
                file += '.png'
            shutil.copyfile(os.path.join(root, 'training', sub, file), os.path.join(root, 'val',sub,file))