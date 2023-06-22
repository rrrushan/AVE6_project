# Automated Driving in Mixed Traffic SoSe-23 (CARISSMA)

Using CARLA and ROS1 Noetic

## This repository contains supplementary files and scripts of Team 2, it includes the following directories:
- `calibration/`
    - `blener/`
    - `Carla/`
- `RVIZ/`
- `test_scenario/`

## Calibration
`calibration/blener/` directory includes the calibration target model. Moreover, it includes the results of experiments preformed in Blender for Team 3 in order to investigate how image modifications influence the intrinsic parameters of the camera. 

`calibration/Carla/` directory contains files used for camera calibration in CARLA and calibration results:
- `Calibration_Images` directory contains images of calibration target generated in CARLA and some images with detected chessboard corners
- `23_05_22.txt` contains the final calibration results
- `calibration_carla.py` is the script to perform camera calibration
- `getting_actor.py` is the script for generation of calibration images in CARLA

`calib_37-26_full_85.txt` file contains results of calibration of the real camera, which was used to determine the real distortion parameters

## RVIZ

`RVIZ/` directory contains RVIZ configuration files necessary to visualize the sensor data published by the CARLA ROS bridge.
- `compare_images.rviz` configuration was used to compare the images recorded during the real test and the images from test scenario recreated in CARLA.
- `dist_rgb_camera_point_cloud.rviz` configuration was used to visualize the RGB camera image, the distorted image and the point cloud data from the ROS bridge.
- `rgb_camera_point_cloud.rviz` configuration was used to visualize the original RGB camera image and the point cloud data.
- `rgb_camera.rviz` configuration was used to visualize only the original RGB camera image.
## Test scenario
`test_scenario` directory contains the scripts necessary for the test scenario in CARLA.
- `calculate_ego.py` computes the coordinates of the spawn point of the ego vehicle
- `find_sensors.py` finds the sensors in CARLA and prints their coordinates to verify that the sensors were placed at the correct coordinates
- `scenario_test.py` was used to create the test scenarios, the weather conditions, type of the target to use and the distance to the target can be controlled with flags inside the script