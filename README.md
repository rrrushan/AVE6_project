# Automated Driving in Mixed Traffic (CARISSMA)
---
Using CARLA and ROS1 Noetic

## DD3D (3D Object Detection)
### Installation
- Install python3.8 in system (python3.9 didnt have appropriate wheels for pytorch3D)
- Follow docker file [Dockerfile](dd3d/docker/Dockerfile-cu111)
- Downgrade protobuf: 
    ```python
    pip uninstall protobuf
    pip install protobuf==3.20.1
    ```
- Install torchinfo (For clean model summaries)
    ```python
    pip install torchinfo
    ```
### Commands
- Training commands:
    ```bash
    python scripts/train.py +experiments=dd3d_kitti_dla34 EVAL_ONLY=True MODEL.CKPT=../dla34_exp.pth TEST.IMS_PER_BATCH=8

    python scripts/train.py +experiments=dd3d_nusc_dla34_custom MODEL.CKPT=/home/carla/admt_student/team3_ss23/AVE6_project/dd3d/depth_pretrained_weights/depth_pretrained_dla34-2lnfuzr1.pth
    ```
- Convert images in folder to video:
    ```bash
    ffmpeg -i /home/carla/admt_student/team3_ss23/ROS_1/bag_imgs/manual_v99_evgresize/%01d.png -r 30 -c:v h264_nvenc out_v99_evg.mp4
    ```
- ROS commands: 
    ```bash
    roscore
    rosbag play ~/admt_student/team3_ss23/ROS_1/bags/2023-04-14-21-04-50.bag -r 0.1 -l
    rosrun rviz rviz
    rosrun my_robot_controller img_subscriber.py
    ```

## Reference
[DD3D Original Repo](https://github.com/TRI-ML/dd3d)
```
@inproceedings{park2021dd3d,
  author = {Dennis Park and Rares Ambrus and Vitor Guizilini and Jie Li and Adrien Gaidon},
  title = {Is Pseudo-Lidar needed for Monocular 3D Object detection?},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  primaryClass = {cs.CV},
  year = {2021},
}