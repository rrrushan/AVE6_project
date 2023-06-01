#!/usr/bin/env python3

from __future__ import print_function

import numpy
import math
import rospy
import cv2 as cv
import json
import numpy as np

import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy

from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge


class DistImagePub(CompatibleNode):
    def __init__(self):
        super(DistImagePub, self).__init__("DistortedImagePublisher")
        self.role_name = self.get_param("role_name", "ego_vehicle")

        # get intrinsic parameters and calculate distortion maps
        self.camera_intrinsics_file = self.get_param('camera_intrinsics_file', '')
        self.image_width, self.image_height, self.image_width_orig, self.image_height_orig, self.K, self.distortion = self.get_intrinsics()
        self.mapx, self.mapy = self.get_dist_maps()
        # create camera info message
        self.camera_info = self.build_camera_info()

        # # use original image
        # self.image_subscriber = self.new_subscription(
        #     Image, "/carla/{}/rgb_front/image".format(self.role_name),
        #     self.dist_image, qos_profile=10)
        
        # use image with additional pixels to get rid of black borders after distortion
        self.image_subscriber = self.new_subscription(
            Image, "/carla/{}/rgb_front_border/image".format(self.role_name),
            self.dist_image, qos_profile=10)
        
        self.image_publisher = self.new_publisher(
            Image,"/carla/{}/rgb_front/image_dist".format(self.role_name),
            qos_profile=10)
        
        self.camera_info_publisher = self.new_publisher(
            CameraInfo,"/carla/{}/rgb_front/camera_info_dist".format(self.role_name),
            qos_profile=10)

    def get_intrinsics(self):
        with open(self.camera_intrinsics_file) as handle:
            camera_intrinsics = json.loads(handle.read())
            image_width = camera_intrinsics["image_width"]
            image_height = camera_intrinsics["image_height"]
            image_width_orig = camera_intrinsics["image_width_orig"]
            image_height_orig = camera_intrinsics["image_height_orig"]
            intr_values = camera_intrinsics["K"]
            K = np.array(([intr_values["fx"], 0.0, intr_values["cx"]], 
                          [0.0, intr_values["fy"], intr_values["cy"]],
                          [0.0, 0.0, 1.0]), dtype='float64')
            dist_values = camera_intrinsics["dist_coef"]
            distortion = np.array((dist_values["k1"], dist_values["k2"], dist_values["p1"], dist_values["p2"], dist_values["k3"]))    
        return image_width, image_height, image_width_orig, image_height_orig, K, distortion
    
    def get_dist_maps(self):
        # calculates maps to distort an image by inverting undistortion maps
        mapx, mapy = cv.initUndistortRectifyMap(self.K, self.distortion, None, None, (self.image_width, self.image_height), 5)
        xymap = np.dstack((mapx, mapy))
        inv_map = self.invert_map(xymap)
        mapx_inv = inv_map[:, :, 0]
        mapy_inv = inv_map[:, :, 1]
        return mapx_inv, mapy_inv
    
    def invert_map(sefl, F):
        # https://github.com/opencv/opencv/issues/22120
        # shape is (h, w, 2), an "xymap"
        (h, w) = F.shape[:2]
        I = np.zeros_like(F)
        I[:,:,1], I[:,:,0] = np.indices((h, w)) # identity map
        P = np.copy(I)
        for i in range(10):
            correction = I - cv.remap(F, P, None, interpolation=cv.INTER_LINEAR)
            P += correction * 0.5
        return P
    
    def build_camera_info(self):
        """
        Function to compute camera info, camera info doesn't change over time
        """
        camera_info = CameraInfo()
        camera_info.width = self.image_width
        camera_info.height = self.image_height
        fx, _, cx, _, fy, cy, _, _, _ = self.K.ravel()
        camera_info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        print(type(camera_info.K[0]))
        camera_info.distortion_model = "brown"
        camera_info.D = self.distortion.tolist()
        return camera_info
    
    def dist_image(self, img_msg):
        """
        Callback when receiving a camera image
        """ 
        header = img_msg.header
        img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")
        dist_img = cv.remap(img, self.mapx, self.mapy, cv.INTER_LINEAR)

        border_x = (self.image_width - self.image_width_orig) //2
        border_y = (self.image_height - self.image_height_orig) //2
        dist_img = dist_img[border_y:-border_y, border_x:-border_x]
 
        # note
        # the ripples in the upper left corner of the distorted uncropped image occur due to the way the inverted map is constructed
        # the appear in the part of the image that should be black

        cam_info = self.camera_info
        img_msg = CvBridge().cv2_to_imgmsg(dist_img)

        cam_info.header = header
        img_msg.header = header
        img_msg.header.frame_id = '{}/rgb_front'.format(self.role_name)

        self.camera_info_publisher.publish(cam_info)
        self.image_publisher.publish(img_msg)


def main(args=None):
    try: 
        roscomp.init("dist_img_pub", args=args)
        dist_node = DistImagePub()
        dist_node.loginfo("Distorted images publisher started")
        dist_node.spin()
    finally: 
        roscomp.shutdown()

if __name__ == "__main__":
    main()