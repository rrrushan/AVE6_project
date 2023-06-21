# Automated Driving in Mixed Traffic SoSe-23 (CARISSMA)

Using CARLA and ROS1 Noetic

![ezgif com-video-to-gif (1)](https://github.com/rrrushan/AVE6_project/assets/75610733/4416ad8e-4422-4b55-b24e-85e775a96faf)

## CARLA ROS bridge 
This repository contains the ROS bridge package with modifications reqiured for recreation of the real test scenarios in CARLA.

## Installation
To use this package the following is required:
- CARLA 0.9.13 or later
- ROS Noetic
- CARLA ROS bridge: https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros1/
- This repository has to be pulled in the same folder as the ROS Bridge

## Modifications
This repository includes the following modifications to ROS bridge:
- the main launch file `ros-bridge/carla_ros_bridge/launch/admt_carla_ros_bridge_with_example_ego_vehicle.launch`
- sensor configuration file `ros-bridge/carla_spawn_objects/config/objects_admt.json`
- `ros-bridge/admt` package, which includes camera distortion node 
- `ros-bridge/traffic_generator` package 

## Launch
```
roslaunch carla_ros_bridge admt_carla_ros_bridge_with_example_ego_vehicle.launch
```
With a CARLA server running on a different PC:
```
roslaunch carla_ros_bridge admt_carla_ros_bridge_with_example_ego_vehicle.launch host:=10.116.80.2
```

The launch file launches the ROS bridge itself, spawns an ego vehicle at a specified spawn point and with the set of sensors listed in the `ros-bridge/carla_spawn_objects/config/objects_admt.json` file, and starts the manual control node. Additionally, it optionally starts the image distortion node and the traffic generation node.

## Sensor configuration
The `objects_admt.json` configuration file includes multiple sensors necessary for the ROS bridge. The sensors necessary for this project:
- `rgb_front` RGB camera sensor configured according to Lucid Vision Labs Triton 2.8 MP  camera parameters
- `lidar_ouster_os1` LiDAR sensor configured according to parameters of Ouster OS1 LiDAR with 128 channels (http://data.ouster.io/downloads/datasheets/datasheet-revd-v2p1-os1.pdf)
- `rgb_front_border` RGB camera with larger image resolution and extended field of view used for image distortion 

## Image distortion
The `admt` package contains the ROS node to perform image distortion in post-processing. The package includes the `camera_dist.launch` file that starts the `camera_dist.py` distortion node, which is utilized by the main launch file. The camera intrinsic parameters and distortion parameters to be used for image distortion are read from the `camera_intrinsics.json` configuration file. 

## Traffic generation
The `traffic_generator` package can be used to populate the CARLA environment with vehicles and pedestrians. The package includes the `generate_traffic.launch` file that starts the `carla_traffic_generator.py` node, which is utilized by the main launch file. At the moment, the input parameters for the traffic generation node, such as CARLA host, and number of vehicles and pedestrians have to be adjusted within the `carla_traffic_generator.py` node itself inside the `GenerateTraffic` class constructor.
