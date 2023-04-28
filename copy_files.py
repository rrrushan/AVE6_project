import os

train_subfile_path = "/media/mobilitylabextreme002/Data1/Videos/nuScenes/v1.0-trainval10_blobs/samples"

target_path = "/media/mobilitylabextreme002/Data1/Videos/nuScenes/samples"

for folder in os.listdir(train_subfile_path):
    print(f"Transferring folder {folder}")
    os.system(f"mv {os.path.join(train_subfile_path, folder)}/* {os.path.join(target_path, folder)}")