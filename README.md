# Automated Driving in Mixed Traffic (CARISSMA)
---
Using CARLA and ROS1 Noetic

## Monocamera 3D Object detection using DD3D
ROS Package to run monocamera object detection and output MarkerArray to visualize on RVIZ

Sample output with Carla Sample in RVIZ
<center><img src="./gifs/carla_cropped_480p.gif" width="400" height="250"/></center>
Sample output with Carissma Sample in RVIZ
<center><img src="./gifs/carissma_cropped_480p.gif" width="400" height="250"/></center>

### Installation
This is a ROS Package with all DD3D files in [src/dd3d](monocam_3D_object_detection/src/dd3d/). So create a catkin workspace and clone this branch in the src/ folder of the workspace:
- Install python3.8 in system (python3.9 didnt have appropriate wheels for pytorch3D, is installed when you follow the below installation commands)
- Install [Nvidia drivers](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/). The testing system has Quadro RTX 5000 (16G) with driver version 520.56.06 with CUDA 11.8.
- The main component instruction follows the docker file [Dockerfile](monocam_3D_object_detection/src/dd3d/docker/Dockerfile-cu111). Here is a simplified version of it:
    - ```bash
      sudo apt-get update && apt-get install -y \
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
      python3.8 \
      python3.8-dev \
      python3-tk \
      python3.8-distutils \
      # opencv
      python3-opencv \
      # set python
      && ln -sf /usr/bin/python3.8 /usr/bin/python \
      && ln -sf /usr/bin/python3.8 /usr/bin/python3 \
      && rm -rf /var/lib/apt/lists/*
      ```
    - Install SSH Server
      ```bash
      sudo apt-get update && apt-get install -y --no-install-recommends openssh-client \ openssh-server && mkdir -p /var/run/sshd
      ```
    - Update pip
      ```bash
      curl -O https://bootstrap.pypa.io/get-pip.py && \
      python get-pip.py && \
      rm get-pip.py
      ```
    - Install Python packages
      ```bash
      python3.8 -m pip install -r requirements.txt
      ```
    - Install PyTorch and other detection related packages
      ```bash
      python3.8 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
      python3.8 -m pip install -U 'git+https://github.com/facebookresearch/fvcore'
      python3.8 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
      python3.8 -m pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu111_pyt190/download.html
      ```
    
- Downgrade protobuf: 
    ```python
    python3.8 -m pip uninstall protobuf
    python3.8 -m pip install protobuf==3.20.1
    ```

### Commands
Model weights are too big to be pushed in GitHub, so they should locally copied into [trained_final_weights folder](monocam_3D_object_detection/src/dd3d/trained_final_weights/) and [depth_pretrained_weights folder](monocam_3D_object_detection/src/dd3d/depth_pretrained_weights)
- **Run Inference on single images in a folder**
    - Open [run_inference.py](monocam_3D_object_detection/src/run_inference.py) and change the variables in __main__() function
    - Run it: `cd monocam_3D_object_detection/src && python run_inference.py`
- **Run ROS:**
    - Create a catkin workspace and build it
    - Duplicate anyone [launch file](monocam_3D_object_detection/src/launch/) and modify as necessary. Three presets are available:
        - carissma.launch for real-world carissma data with RGB and ouster LiDAR
        - carla_sample.launch for sample carla .rosbags
        - carla_testTrack.launch for carla .rosbags actually used for evaluating Digital Twin
    - **Carefully modify the paths and parameters as required in the .launch file. For both detection node and `rosbag play` node**
    - To run: `roslaunch monocam_3D_object_detection <file_name>.launch`
- Training commands:
    ```bash
    cd monocam_3D_object_detection/src/dd3d
    python scripts/train.py +experiments=dd3d_kitti_dla34 EVAL_ONLY=True MODEL.CKPT=../dla34_exp.pth TEST.IMS_PER_BATCH=8

    python scripts/train.py +experiments=dd3d_nusc_dla34_custom MODEL.CKPT=/home/carla/admt_student/team3_ss23/AVE6_project/dd3d/depth_pretrained_weights/depth_pretrained_dla34-2lnfuzr1.pth
    ```
- Convert images in folder to video:
    ```bash
    ffmpeg -i /home/carla/admt_student/team3_ss23/ROS_1/bag_imgs/manual_v99_evgresize/%01d.png -r 30 -c:v h264_nvenc out_v99_evg.mp4
    ```

## Benchmark
Time for transform + inference in FP16
-> Insert from Vishal PC
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