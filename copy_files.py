import os

train_subfile_path = "/home/carla/admt_student/team3_ss23/data/NuScenes/v1.0-trainval03_blobs_camera/samples"

target_path = "/home/carla/admt_student/team3_ss23/data/NuScenes/samples"

for folder in os.listdir(train_subfile_path):
    print(f"Transferring folder {folder}")
    # os.system(f"mv {os.path.join(train_subfile_path, folder)}/* {os.path.join(target_path, folder)}")