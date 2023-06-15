#!/usr/bin/env python  

"""
Generate /tf and /camera_info topic for real camera .rosbags.
These are required for proper visualization in RVIZ.

The intrinsic parameters for camera are obtained from checkerboard calibration.
For /tf, we require actual positions of camera/LiDAR along with rotations. Since rotation angles,
was not recorded during test, some offsets of bounding box position/orientation could be observed.
This offset has been neutralized by trial-and-error method by reducing height (z-axis) of the camera.

Camera image and LiDAR pointcloud are also republished to have uniform timestamps among 
all topics.
"""

from sensor_msgs.msg import CompressedImage, PointCloud2, CameraInfo
import rospy
import tf
import numpy as np

# Camera: /arena_camera_node/image_raw/compressed
# LiDAR: /ouster/points

class TF_CamInfo_Carissma:
    def __init__(self):
        rospy.init_node('carissma_tf_caminfo_broadcaster')
        
        self.cam_info = CameraInfo()
        self.cam_info.height = 1464
        self.cam_info.width = 1936
        self.cam_info.distortion_model = "plumb_bob"
        self.cam_info.D = [-0.27529969,  0.05905619, -0.00245923,  0.00061132,  0.11663574]
        self.cam_info.K = [1.33775968e+03, 0.00000000e+00, 9.67509657e+02,
                        0.00000000e+00, 1.33568429e+03, 7.33902189e+02,
                        0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        self.cam_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.cam_info.P = [1.33775968e+03, 0.00000000e+00, 9.67509657e+02, 0.0,
                        0.00000000e+00, 1.33568429e+03, 7.33902189e+02, 0.0,
                        0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.0]
        
        sub_lidar = rospy.Subscriber(
            '/ouster/points',
            PointCloud2,
            self.callback_lidar
        )
        sub_camera = rospy.Subscriber(
            '/arena_camera_node/image_raw/compressed',
            CompressedImage,
            self.callback_camera
        )

        self.pub_lidar = rospy.Publisher(
            "/ouster/points_adapted",
            PointCloud2,
            queue_size=30
        )
        self.pub_camera = rospy.Publisher(
            '/adapted_arena_camera/compressed',
            CompressedImage,
            queue_size=30
        )
        self.pub_camera_info = rospy.Publisher(
            '/camera_info',
            CameraInfo,
            queue_size=30
        )
        
        rospy.spin()

    def callback_lidar(self, msg):
        br = tf.TransformBroadcaster()
        msg.header.stamp = rospy.Time.now()
        br.sendTransform((0.0, 0.0, 1.6),
                        (0.0, 0.0, 0.0, 1.0),
                        msg.header.stamp,
                        msg.header.frame_id,
                        "world")
        self.pub_lidar.publish(msg)

    def callback_camera(self, msg):
        br = tf.TransformBroadcaster()
        msg.header.stamp = rospy.Time.now()
        quat = [-0.52, 0.52, -0.47, 0.47] # These quartenions were observed from "ego_vehicle/rgb_front" from carla
        quat = quat / np.linalg.norm(quat)
        br.sendTransform((0.0, 0.0, -0.45), # It has to be actually (0, 0, 1.0) but to neutralize offset, it has been adjusted accordingly
                        quat, 
                        msg.header.stamp,
                        msg.header.frame_id,
                        "world")
        self.cam_info.header = msg.header
        self.pub_camera_info.publish(self.cam_info)
        self.pub_camera.publish(msg)

if __name__ == '__main__':
    TF_CamInfo_Carissma()