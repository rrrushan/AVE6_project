import rosbag
import os
from cv_bridge import CvBridge
import cv2
import tqdm

bridge = CvBridge()

# path_to_bag = '/home/carla/admt_student/team3_ss23/ROS_1/bags/23_04_14/23_04_14_traffic_manual.bag'
# write_folder = "/home/carla/admt_student/team3_ss23/ROS_1/bag_imgs/manual"

path_to_bag = "/media/carla/7E49D24E7314923E/Calibration/2023-05-05-09-40-38.bag"
# 2023-05-05-09-37-26.bag
# 2023-05-05-09-40-38.bag
# 2023-05-05-09-57-48.bag
write_folder = "/home/carla/admt_student/team3_ss23/data/carissma/calibration/2023-05-05-09-37-26"

bag = rosbag.Bag(path_to_bag)
print(bag)

# for index, msg in enumerate(bag.read_messages(topics=['/carla/ego_vehicle/rgb_front/image'])):
# Camera topic:  '/arena_camera_node/image_raw/compressed'
# IR topic: '/flir_adk/image_raw/compressed'

# list_of_strings = []
# for index, msg in tqdm.tqdm(enumerate(bag.read_messages(topics=['/arena_camera_node/image_raw/compressed']))):
#     # print(f"Writing {index} image")
#     #print(msg[2])
#     save_string = f'{str(index)}.png: {msg[2]}'
#     # print(save_string)
#     list_of_strings.append(save_string)
#     # cv2.imwrite(os.path.join(write_folder, f"{index}.png"), bridge.imgmsg_to_cv2(msg[1], "bgr8"))
    
#     # cv2.imwrite(os.path.join(write_folder, f"{index}.png"), bridge.compressed_imgmsg_to_cv2(msg[1], "bgr8"))

    
# with open('2023-05-05-09-40-38_cam_timestamp.txt', 'w') as f:
#     # f.writelines(list_of_strings)
#     f.write('\n'.join(list_of_strings))