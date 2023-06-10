#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import MarkerArray, Marker

import numpy as np
from cv_bridge import CvBridge
import cv2

import os
import torch

from torchinfo import summary
import numpy as np
import cv2
from time import perf_counter

import detectron2.utils.comm as d2_comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluators, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, get_event_storage
import omegaconf
import sys

import dd3d.tridet.modeling  # pylint: disable=unused-import
import dd3d.tridet.utils.comm as comm
from dd3d.tridet.data import build_test_dataloader, build_train_dataloader
from dd3d.tridet.data.dataset_mappers import get_dataset_mapper
from dd3d.tridet.data.datasets import random_sample_dataset_dicts, register_datasets
from dd3d.tridet.evaluators import get_evaluator
from dd3d.tridet.modeling import build_tta_model
from dd3d.tridet.utils.s3 import sync_output_dir_s3
from dd3d.tridet.utils.setup import setup
from dd3d.tridet.utils.train import get_inference_output_dir, print_test_results
from dd3d.tridet.utils.visualization import mosaic, save_vis
from dd3d.tridet.utils.wandb import flatten_dict, log_nested_dict
from dd3d.tridet.visualizers import get_dataloader_visualizer, get_predictions_visualizer

class DD3D:
    def __init__(self, cfg):
        # Create model 
        # TODO: Convert to param Caro
        cfg.DD3D.FCOS2D.INFERENCE.PRE_NMS_THRESH = rospy.get_param('~pre_nms_thresh') # 0.1
        cfg.DD3D.FCOS2D.INFERENCE.NMS_THRESH = rospy.get_param('~nms_thresh')         # 0.2 
        # --
        self.model = build_model(cfg)
        checkpoint_file = cfg.MODEL.CKPT
        
        os.chdir("../../..") # Moving cwd from dd3d/outputs/date/time to dd3d/
        
        # Load trained weights
        if checkpoint_file:
            self.model.load_state_dict(torch.load(checkpoint_file)["model"])
        summary(self.model) # Print model summary

        self.model.eval() # Inference mode


        # Params for cropping and rescaling
        self.ORIG_IMG_HEIGHT = rospy.get_param('~ORIG_IMG_HEIGHT')  # 1464 
        self.ORIG_IMG_WIDTH = rospy.get_param('~ORIG_IMG_WIDTH')    # 1936 
        self.TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP = rospy.get_param('~TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP')
        self.CROPPED_IMG_HEIGHT = self.ORIG_IMG_HEIGHT - self.TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP
        target_resize_width = rospy.get_param('~TARGET_RESIZE_WIDTH')
        self.aspect_ratio = target_resize_width / self.ORIG_IMG_WIDTH

        self.TARGET_RESIZE_RES = (target_resize_width, int(self.aspect_ratio*self.CROPPED_IMG_HEIGHT))
        # --

        # Color code for visualizing each class
        # BGR (inverted RGB) color values for classes
        self.color_mapping = {
            0: ((0, 0, 255), "Car"),         # Car: Red
            1: ((0, 255, 0), "Pedestrian"),  # Pedestrian: Green
            2: ((255, 0, 0), "Cyclist"),     # Cyclist: Blue
            3: ((0, 192, 255), "Van"),       # Van: Yellow
            4: ((255, 255, 0), "Truck"),     # Truck: Turquoise Blue
            5: ((0, 0, 0), "Unknown")        # Unknown: Black
        }

        self.car_min_trackwidth = rospy.get_param('~min_trackwidth')
        self.car_max_trackwidth = rospy.get_param('~max_trackwidth')
        self.car_min_wheelbase  = rospy.get_param('~min_wheelbase')
        self.car_max_wheelbase  = rospy.get_param('~max_wheelbase')
        self.car_min_height     = rospy.get_param('~car_min_height')
        self.car_max_height     = rospy.get_param('~car_max_height')

        self.ped_max_base   = rospy.get_param('~max_base')
        self.ped_min_height = rospy.get_param('~min_height')
        self.ped_max_height = rospy.get_param('~max_height')

        
    def load_cam_mtx(self, cam_mtx):
        """Load camera matrix from /camera_info topic and modify with required padding/resizing

        Args:
            cam_mtx (torch.FloatTensor): Original Camera matrix
        """

        self.orig_cam_intrinsic_mtx = cam_mtx
        cam_mtx[0] *= self.aspect_ratio # self.TARGET_RESIZE_RES[0]/self.ORIG_IMG_WIDTH
        cam_mtx[1][2] = cam_mtx[1][2] - self.TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP
        cam_mtx[1] *= self.aspect_ratio
        self.cam_intrinsic_mtx = torch.FloatTensor(cam_mtx)

    def output2MarkerArray(self, predictions, header):
        """Converts model outputs to marker array format for RVIZ

        Args:
            predictions (list[Instance]): 
                Each item in the list has type Instance. Has all detected data
        Returns:
            marker_list (list): List of Markers for RVIZ
        """

        def get_quaternion_from_euler(roll, pitch, yaw):
            """
            Convert an Euler angle to a quaternion.

            Input
                :param roll: The roll (rotation around x-axis) angle in radians.
                :param pitch: The pitch (rotation around y-axis) angle in radians.
                :param yaw: The yaw (rotation around z-axis) angle in radians.

            Output
                :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
            """
            qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
            qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
            qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            
            return qx, qy, qz, qw

        bboxes_per_image = predictions[0]["instances"].pred_boxes3d.corners.detach().cpu().numpy()
        classes_per_image = predictions[0]["instances"].pred_classes.detach().cpu().numpy()
        marker_list = MarkerArray() # []
        
        vis_num = 0 
        for index, (single_bbox, single_class) in enumerate(zip(bboxes_per_image, classes_per_image)):
            class_id = int(single_class)
            if class_id > 1: # Car class: 0, Pedestrian class: 1. Ignoring all the other classes
                continue

            vis_num += 1

            marker_msg = Marker()
            # BBOX is stored as (8, 3) -> where last_index is (y, z, x)
            min_x, max_x = np.min(single_bbox[:, 0]), np.max(single_bbox[:, 0])
            min_y, max_y = np.min(single_bbox[:, 1]), np.max(single_bbox[:, 1])
            min_z, max_z = np.min(single_bbox[:, 2]), np.max(single_bbox[:, 2])

            # Center points of BBOX along each axis
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            cz = (min_z + max_z) / 2
            
            if cz <= 1.5:
                continue

            # Limiting size of BBOX in each axis according parameters in ROS Launch
            if class_id == 0: # Car
                scale_x = np.clip(abs(single_bbox[0, 0] - single_bbox[1, 0]), self.car_min_trackwidth, self.car_max_trackwidth) 
                # scale_x = np.clip(abs(max_x - min_x), self.car_min_trackwidth, self.car_max_trackwidth) 
                scale_y = np.clip(max_y - min_y, self.car_min_height, self.car_max_height)
                scale_z = np.clip(max_z - min_z, self.car_min_wheelbase, self.car_max_wheelbase)

            elif class_id == 1: # Pedestrian
                scale_x = np.clip(abs(max_x - min_x), 0.0, self.ped_max_base) 
                scale_y = np.clip(max_y - min_y, self.ped_min_height, self.ped_max_height)
                scale_z = np.clip(max_z - min_z, 0.0, self.ped_max_base)
            

            ## Rotation of BBOX along each axis
            # Setting Roll and Pitch angle to 0.0 as they the vehicles are considered to be on flat surface.
            # To further reduce noisy bbox positions/scales
            pitch_angle = 0.0 # np.math.atan2((single_bbox[4, 1] - single_bbox[0, 1]), (single_bbox[4, 2] - single_bbox[0, 2]),) # pitch in camera
            roll_angle = 0.0 # np.math.atan2((single_bbox[1, 1] - single_bbox[0, 1]), (single_bbox[1, 0] - single_bbox[0, 0]))
            yaw_angle = -np.math.atan2((single_bbox[1, 2] - single_bbox[0, 2]), (single_bbox[1, 0] - single_bbox[0, 0])) # yaw in camera.
            
            qx, qy, qz, qw = get_quaternion_from_euler(pitch_angle, yaw_angle, roll_angle)
            
            marker_msg.type = Marker.CUBE
            marker_msg.header.stamp = header.stamp
            marker_msg.header.frame_id = "ego_vehicle/rgb_front"
            
            marker_msg.pose.position.x = cx     # in camera frame: y, left-right
            marker_msg.pose.position.y = cy     # in camera frame: z, height
            marker_msg.pose.position.z = cz     # in camera frame: x, depth 
            marker_msg.pose.orientation.x = qx 
            marker_msg.pose.orientation.y = qy 
            marker_msg.pose.orientation.z = qz 
            marker_msg.pose.orientation.w = qw 
            
            
            marker_msg.scale.x = scale_x # Trackwidth
            marker_msg.scale.y = scale_y # Height
            marker_msg.scale.z = scale_z # Wheelbase
            
            """
            # first angle -> pitch, yaw, roll
            qx, qy, qz, qw = get_quaternion_from_euler(np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(10.0))
            marker_msg.pose.position.x = 3.0     # in camera frame: y, left-right
            marker_msg.pose.position.y = 1.0     # in camera frame: z. height
            marker_msg.pose.position.z = 10.0    # in camera frame: x, depth
            marker_msg.pose.orientation.x = qx #- 0.5
            marker_msg.pose.orientation.y = qy #+ 0.5
            marker_msg.pose.orientation.z = qz #- 0.5
            marker_msg.pose.orientation.w = qw #+ 0.5
            
            marker_msg.scale.x = 2.0 # scale_x, trackwidth
            marker_msg.scale.y = 1.0 # scale_y. height of the car
            marker_msg.scale.z = 4.0 # scale_z, wheelbase
            """

            color               = self.color_mapping[class_id][0]
            marker_msg.color.r  = color[2]
            marker_msg.color.g  = color[1]
            marker_msg.color.b  = color[0]
            marker_msg.color.a  = 0.5
            marker_msg.id       = vis_num
            marker_msg.lifetime = rospy.Duration()
            
            marker_list.markers.append(marker_msg)

            marker_msg = Marker()
            
        return marker_list

    def transform_img(self, img):
        # Crop off top pixels to remove unnecessary image
        # cv2.imwrite("/home/carla/admt_student/team3_ss23/ROS_Project/src/monocam_3D_object_detection/Orign_img.png", img)
        img = img[self.TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP:img.shape[0], :]
        # cv2.imwrite("/home/carla/admt_student/team3_ss23/ROS_Project/src/monocam_3D_object_detection/Cropped_img.png", img)
        # Resize Image to target resolution
        img = cv2.resize(img, self.TARGET_RESIZE_RES)
        # cv2.imwrite("/home/carla/admt_student/team3_ss23/ROS_Project/src/monocam_3D_object_detection/Resized_img.png", img)
        # exit()
        return img

    def rescale_boxes(self, boxes):
        """Rescale boxes after cropping and padding

        Args:
            boxes (np.array): 3D boxes. Shape: [Batchsize, 8, 2]

        Returns:
            final_box_points (np.array): Modified 3D boxes. Shape: [Batchsize, 8, 2]
        """
       
        remove_resize_x, remove_resize_y = boxes[:, :, 0] * 1 / self.aspect_ratio, boxes[:, :, 1] * 1 / self.aspect_ratio 
        remove_resize_y = remove_resize_y + self.TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP
        final_box_points = np.dstack((remove_resize_x, remove_resize_y)).astype(int)

        return final_box_points
    
    def visualize(self, batched_images, batched_model_output, mode="3D", rescale=True):
        """Draws the bounding box along with the class name and confidence score
        on the given images. The list in input and output refers to batching of data.

        Args:
            batched_images (list[np.array]): 
                List of OpenCV images
            batched_model_output (list[Instance]): 
                Each item in the list has type Instance. Has all detected data
            mode (str, optional): 
                "3D" or "2D" for bbox visualization. Defaults to "3D".

        Raises:
            ValueError: If wrong value for "mode" kwarg is given

        Returns:
            final_images (list[np.array]): 
                List of all visualized images
        """
        final_images = []
        for single_output, image in zip(batched_model_output, batched_images):
            pred_classes = single_output["instances"].pred_classes.detach().cpu().numpy() # ClassIDs of detections
            pred_scores = single_output["instances"].scores_3d.detach().cpu().numpy() # Confidence scores
            pred_bbox3d = single_output["instances"].pred_boxes3d.corners.detach().cpu().numpy() # 3D corners

            ## Project 3D world points to image
            # Multiply the 3D points by the intrinsic matrix to obtain the 2D points
            points_2d_homogeneous = np.dot(pred_bbox3d, self.cam_intrinsic_mtx.cpu().numpy().T) # intrinsic_matrix.dot(points_3d.T).T
            # Convert the homogeneous 2D points to non-homogeneous coordinates
            pred_bbox3d_img = points_2d_homogeneous[:, :, :2] / points_2d_homogeneous[:, :, 2:]

            if rescale:
                pred_bbox3d_img = self.rescale_boxes(pred_bbox3d_img)

            if mode == "3D":
                # Define the edges of the bounding box.
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ]
                img_alpha = image.copy()

                for box, classID, conf_score in zip(pred_bbox3d_img, pred_classes, pred_scores):
                    if classID > 1: # Only car and pedestrian class
                        continue

                    class_color = self.color_mapping[int(classID)][0]
                    class_name = self.color_mapping[int(classID)][1]

                    # Draw the edges of the 3D bounding box
                    for edge in edges:
                        start_point = tuple(box[edge[0]].astype(int))
                        end_point = tuple(box[edge[1]].astype(int))
                        cv2.line(
                            image, start_point, end_point, 
                            class_color, 2, cv2.LINE_AA
                        )
                    
                    # Plot class name and score
                    baseLabel = f'{class_name} {round(conf_score*100)}%'
                    (w1, h1), _ = cv2.getTextSize(
                        baseLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1
                    )
                    
                    # Calculating the right coordinates to place the text
                    # in the bottom edge of the bbox
                    min_bottom_x = min(box[2][0], box[3][0])
                    max_bottom_x = max(box[2][0], box[3][0])
                    bottom_edge_width = max_bottom_x - min_bottom_x
                    gap = (bottom_edge_width - w1) / 2
                    text_location_x1 = int(min_bottom_x + gap)
                    text_location_y1 = int((box[2][1] + box[3][1]) / 2)
                    
                    # Displays BG Box for the text and text itself
                    cv2.rectangle(
                        image, (text_location_x1, text_location_y1),
                        (text_location_x1 + w1, text_location_y1 + h1 + 1), class_color,
                        -1, cv2.LINE_AA
                    )
                    image = cv2.putText(
                        image, baseLabel, (text_location_x1, text_location_y1 + h1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA
                    )  
                    # Fill just the front face
                    points = np.array([
                        tuple(box[0].astype(int)), tuple(box[1].astype(int)), 
                        tuple(box[2].astype(int)), tuple(box[3].astype(int))
                    ])
                    
                    # Use fillPoly() function and give input as image,
                    cv2.fillPoly(img_alpha, pts=[points], color=class_color)

                # To create transparency effect of the shaded box
                blend = cv2.addWeighted(image, 0.75, img_alpha, 0.25, 0)
                final_images.append(blend)
            
            elif mode == "2D":
                img_2D = image.copy()
                for box, classID in zip(pred_bbox3d_img, pred_classes):
                    class_color = self.color_mapping[int(classID)][0]
                    
                    min_x, min_y = np.amin(box, 0)
                    max_x, max_y = np.amax(box, 0)
                    first_point = (int(min_x), int(min_y))
                    second_point = (int(max_x), int(max_y))

                    cv2.rectangle(
                        img_2D, 
                        first_point, 
                        second_point, 
                        class_color
                    )
                final_images.append(img_2D)
            else:
                raise ValueError(f"Only modes 3D or 2D is accepted for visualization. Given: {mode}")
        return final_images
    
    def inference_on_single_image(self, image):
        """Runs inference on single image

        Args:
            image (np.array): 
                OpenCV image format (H, W, C)

        Returns:
            output (list[Instance]): 
                Each item in the list has type Instance. Has all detected data
        """
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        transformed_img = torch.from_numpy(image)
        
        # Transposing: [H, W, C] -> [C, H, W] (KiTTi: [3, 375, 1242])
        transformed_img = transformed_img.permute(2, 0, 1)

        with torch.cuda.amp.autocast():
            output = self.model.predict(transformed_img[None, :], [self.cam_intrinsic_mtx])

        return output

class obj_detection:
    def __init__(self):
        rospy.init_node("img_subscriber", anonymous= True)
        # sys.argv.append('+experiments=dd3d_kitti_dla34')
        # sys.argv.append('MODEL.CKPT=/home/carla/admt_student/team3_ss23/AVE6_project/dd3d/trained_final_weights/dla34.pth')

        # cfg = omegaconf.OmegaConf.load("/home/carla/admt_student/team3_ss23/AVE6_project/dd3d/trained_final_weights/dla.yaml")
        # cfg.MODEL.CKPT = "/home/carla/admt_student/team3_ss23/AVE6_project/dd3d/trained_final_weights/dla34.pth"

        # name_path = "/home/carla/admt_student/team3_ss23/AVE6_project/dd3d/trained_final_weights/v99.yaml"
        #cfg = omegaconf.OmegaConf.load(name_path)
        #cfg = omegaconf.OmegaConf.load("/home/carla/admt_student/team3_ss23/AVE6_project/dd3d/trained_final_weights/v99.yaml") # v99.yaml
        cfg = omegaconf.OmegaConf.load(rospy.get_param('~config_path'))

        #cfg.MODEL.CKPT = "/home/carla/admt_student/team3_ss23/AVE6_project/dd3d/trained_final_weights/v99.pth" # v99.pth
        cfg.MODEL.CKPT = rospy.get_param('~model_checkpoint_path')
        self.dd3d = DD3D(cfg)
        # --

        

        # Wait for camera intrinsic data
        rospy.loginfo("Waiting for /camera_info topic")
        cam_info_data = rospy.wait_for_message(rospy.get_param('~camera_info_topic'), CameraInfo)
        cam_intrinsic_mtx = np.reshape(np.array(cam_info_data.K), (3,3))
        self.dd3d.load_cam_mtx(cam_intrinsic_mtx)

        sub_image = rospy.Subscriber(rospy.get_param('~camera_img_topic'), Image, callback=self.callback_image)
        self.marker_pub = rospy.Publisher(rospy.get_param('~marker_output_topic'), MarkerArray, queue_size=10)
        
        self.vis_image_pub = rospy.Publisher("rgb_front/vis_image", Image, queue_size=10)
        rospy.loginfo("img_sub has been created")
        
        self.cam_intrinsic_mtx = None
        self.cvbridge = CvBridge()

        # Checking for compressed ROS image (used by camera recorded rosbags, not for carla)
        if rospy.get_param("~use_compressed_image"):
            self.ros2cv2 = self.cvbridge.compressed_imgmsg_to_cv2
        else:
            self.ros2cv2 = self.cvbridge.imgmsg_to_cv2
        self.enable_visualization = rospy.get_param("~debug_enable_intern_vis")
        rospy.spin()


    def callback_image(self, msg: Image):
        header_info = msg.header
        t_1 = perf_counter()
        cv_image = self.ros2cv2(msg, "bgr8")
        transform_img = self.dd3d.transform_img(cv_image)
        t_2 = perf_counter()
        predictions = self.dd3d.inference_on_single_image(transform_img)
        t_3 = perf_counter()
        marker_list = self.dd3d.output2MarkerArray(predictions, header_info)
        self.marker_pub.publish(marker_list)
        t_4 = perf_counter()

        total_time = (t_4 - t_1)*1000
        if self.enable_visualization:
            final_image = self.dd3d.visualize([cv_image], predictions)[0]
            ros_img = self.cvbridge.cv2_to_imgmsg(final_image)

            self.vis_image_pub.publish(ros_img)
            rospy.loginfo(f"MonoCam3D Obj.Det| Shape: {self.dd3d.TARGET_RESIZE_RES}, Time log| Total: {total_time:.2f}ms, Img_Tf: {(t_2 - t_1)*1000:.2f}ms, Inference: {(t_3 - t_2)*1000:.2f}ms, conv_toMarker: {(t_4 - t_3)*1000:.2f}ms")
        else:
            rospy.loginfo(f"MonoCam3D Obj.Det| Time per frame: {total_time:.2f}ms, {1000/total_time :.2f}fps")

if __name__ == '__main__':
    random_obj = obj_detection()