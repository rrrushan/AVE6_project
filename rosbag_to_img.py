import rosbag
import os
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()

# path_to_bag = '/home/carla/admt_student/team3_ss23/ROS_1/bags/23_04_14/23_04_14_traffic_manual.bag'
# write_folder = "/home/carla/admt_student/team3_ss23/ROS_1/bag_imgs/manual"

path_to_bag = "/home/carla/Downloads/2023-04-21-11-03-33.bag"
write_folder = "/home/carla/admt_student/outputs/team3"

bag = rosbag.Bag(path_to_bag)

# for index, msg in enumerate(bag.read_messages(topics=['/carla/ego_vehicle/rgb_front/image'])):
for index, msg in enumerate(bag.read_messages(topics=['/arena_camera_node/image_raw'])):
    print(f"Writing {index} image")
    cv2.imwrite(os.path.join(write_folder, f"{index}.png"), bridge.imgmsg_to_cv2(msg[1], "bgr8"))
    