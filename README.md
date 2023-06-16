# Automated Driving in Mixed Traffic (CARISSMA)
---
Using CARLA and ROS1 Noetic

## DD3D (3D Object Detection)
### Installation
- Install python3.8 in system (python3.9 didnt have appropriate wheels for pytorch3D)
- Install [Nvidia drivers](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/). The testing system has Quadro RTX 5000 (16G) with driver version 520.56.06 with CUDA 11.8.
- The main component instruction follows the docker file [Dockerfile](monocam_3D_object_detection/src/dd3d/docker/Dockerfile-cu111). Here is a simplified version of it:
    - ```bash
      apt-get update && apt-get install -y \
      # essential
      build-essential \
      cmake \
      ffmpeg \
      g++-4.8 \
      git \
      curl \
      docker.io \
      vim \
      wget \
      unzip \
      htop \
      libjpeg-dev \
      libpng-dev \
      libavdevice-dev \
      pkg-config \
      # python
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python3-tk \
      python${PYTHON_VERSION}-distutils \
      # opencv
      python3-opencv \
      # set python
      && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
      && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
      && rm -rf /var/lib/apt/lists/*
      ```
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
## Benchmark
Time for transform + inference in FP16
- V99
Target Img Res: (144, 480, 3), total time: 0.063s, transform time: 0.002s, model time: 0.061s
Target Img Res: (217, 720, 3), total time: 0.066s, transform time: 0.003s, model time: 0.063s
Target Img Res: (289, 960, 3), total time: 0.069s, transform time: 0.003s, model time: 0.066s
Target Img Res: (386, 1280, 3), total time: 0.082s, transform time: 0.003s, model time: 0.079s
Target Img Res: (483, 1600, 3), total time: 0.101s, transform time: 0.003s, model time: 0.098s
Target Img Res: (579, 1920, 3), total time: 0.134s, transform time: 0.003s, model time: 0.131s

- DLA34
Target Img Res: (144, 480, 3), total time: 0.046s, transform time: 0.002s, model time: 0.044s
Target Img Res: (217, 720, 3), total time: 0.046s, transform time: 0.003s, model time: 0.043s
Target Img Res: (289, 960, 3), total time: 0.048s, transform time: 0.003s, model time: 0.045s
Target Img Res: (386, 1280, 3), total time: 0.048s, transform time: 0.003s, model time: 0.045s
Target Img Res: (483, 1600, 3), total time: 0.049s, transform time: 0.003s, model time: 0.047s
Target Img Res: (579, 1920, 3), total time: 0.059s, transform time: 0.003s, model time: 0.056s

- OmniML
Target Img Res: (144, 480, 3), total time: 0.050s, transform time: 0.002s, model time: 0.047s
Target Img Res: (217, 720, 3), total time: 0.053s, transform time: 0.003s, model time: 0.050s
Target Img Res: (289, 960, 3), total time: 0.053s, transform time: 0.003s, model time: 0.050s
Target Img Res: (386, 1280, 3), total time: 0.054s, transform time: 0.003s, model time: 0.052s
Target Img Res: (483, 1600, 3), total time: 0.056s, transform time: 0.003s, model time: 0.053s
Target Img Res: (579, 1920, 3), total time: 0.054s, transform time: 0.003s, model time: 0.051s

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