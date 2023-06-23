#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

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
        try:
            cfg.DD3D.FCOS2D.INFERENCE.PRE_NMS_THRESH = rospy.get_param('~pre_nms_thresh') # 0.1
            cfg.DD3D.FCOS2D.INFERENCE.NMS_THRESH = rospy.get_param('~nms_thresh')         # 0.2 
            
            self.model = build_model(cfg)
        except:
            rospy.logerr("MonoCam3D Obj.Det| You had one job.....CHECK THE MODEL .cfg PATHS !!!!")
            rospy.signal_shutdown()
        rospy.loginfo("MonoCam3D Obj.Det| Model config (.cfg) loaded successfully")

        try:
            checkpoint_file = cfg.MODEL.CKPT
            os.chdir("../../..") # Moving cwd from dd3d/outputs/date/time to dd3d/
            
            # Load trained weights
            if checkpoint_file:
                self.model.load_state_dict(torch.load(checkpoint_file)["model"])
        except:
            rospy.logerr("MonoCam3D Obj.Det| No bueno.....CHECK THE MODEL .pth (checkpoint) PATHS !!!!")
            rospy.signal_shutdown()
        rospy.loginfo("MonoCam3D Obj.Det| Model checkpoint (.pth) loaded successfully")

        summary(self.model) # Print model summary
        self.model.eval() # Inference mode
        # --

        # Specific params to differentiate real_camera .rosbags and carla .rosbags
        self.rviz_cam_output_frame = rospy.get_param('~rviz_cam_output_frame')
        self.realcam_capture = rospy.get_param('~realcam_capture')
        self.debug_enable_visualization = rospy.get_param("~debug_enable_intern_vis")
        self.rviz_marker_retain_duration = rospy.get_param('~rviz_marker_retain_duration')

        # Params for cropping and rescaling
        self.ORIG_IMG_HEIGHT = rospy.get_param('~ORIG_IMG_HEIGHT')  # 1464 
        self.ORIG_IMG_WIDTH = rospy.get_param('~ORIG_IMG_WIDTH')    # 1936 
        self.TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP = rospy.get_param('~TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP')
        self.CROPPED_IMG_HEIGHT = self.ORIG_IMG_HEIGHT - self.TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP
        target_resize_width = rospy.get_param('~TARGET_RESIZE_WIDTH')
        self.aspect_ratio = target_resize_width / self.ORIG_IMG_WIDTH
        self.TARGET_RESIZE_RES = (target_resize_width, int(self.aspect_ratio*self.CROPPED_IMG_HEIGHT))

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

        ## Object filteration parameters - Objects lying beyond this range are not visualized
        # Car
        self.car_min_trackwidth = rospy.get_param('~min_trackwidth')
        self.car_max_trackwidth = rospy.get_param('~max_trackwidth')
        self.car_min_wheelbase  = rospy.get_param('~min_wheelbase')
        self.car_max_wheelbase  = rospy.get_param('~max_wheelbase')
        self.car_min_height     = rospy.get_param('~car_min_height')
        self.car_max_height     = rospy.get_param('~car_max_height')

        # Pedestrian and Cyclist
        self.ped_max_base   = rospy.get_param('~max_base')
        self.ped_min_height = rospy.get_param('~min_height')
        self.ped_max_height = rospy.get_param('~max_height')

        
    def load_cam_mtx(self, cam_mtx):
        """Load camera matrix from /camera_info topic and modify with required padding/resizing

        Args:
            cam_mtx (torch.FloatTensor): Original Camera matrix
        """

        self.orig_cam_intrinsic_mtx = cam_mtx
        cam_mtx[0] *= self.aspect_ratio
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

        # Create a marker object
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (0, 2), (1, 3)
        ]
        bboxes_per_image = predictions[0]["instances"].pred_boxes3d.corners.detach().cpu().numpy()
        classes_per_image = predictions[0]["instances"].pred_classes.detach().cpu().numpy()
        marker_list = MarkerArray() # []

        vis_num = 0
        for single_bbox, single_class in zip(bboxes_per_image, classes_per_image):
            class_id = int(single_class)
            if class_id > 2: # Car class: 0, Pedestrian class: 1, Bike class: 2. Ignoring all the other classes
                continue

            vis_num += 1
            marker_msg = Marker()
            
            # BBOX is stored as (8, 3) -> where last_index is (y, z, x). For cam coord: (0, 1, 2), (2, 0, 1)
            min_x, max_x = np.min(single_bbox[:, 0]), np.max(single_bbox[:, 0])
            min_y, max_y = np.min(single_bbox[:, 1]), np.max(single_bbox[:, 1])
            min_z, max_z = np.min(single_bbox[:, 2]), np.max(single_bbox[:, 2])

            # Center points of BBOX along each axis
            cx = (min_x + max_x) / 2 # Height from the ground
            cy = (min_y + max_y) / 2 # Left-right distance from camera
            cz = (min_z + max_z) / 2 # Depth of the object straightahead
            distance_of_object = np.linalg.norm([cx, cz])
            
            yaw_angle = -np.math.atan2((single_bbox[1, 2] - single_bbox[0, 2]), (single_bbox[1, 0] - single_bbox[0, 0]))
            if distance_of_object <= 1.5: # For boxes too close to the camera
                continue
            
            vis_num += 1
            marker = Marker()
            marker.header.frame_id = self.rviz_cam_output_frame
            marker.header.stamp = header.stamp
            marker.ns = self.color_mapping[class_id][1]
            marker.id = vis_num
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.1 # Line width

            color          = self.color_mapping[class_id][0]
            marker.color.r = color[2] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[0] / 255.0
            marker.color.a = 1.0

            # Add the each pair of consecutive points to form an edge
            for edge in edges:
                point = Point()
                point.x = single_bbox[edge[0]][0]
                point.y = single_bbox[edge[0]][1]
                point.z = single_bbox[edge[0]][2]
                marker.points.append(point)

                point = Point()
                point.x = single_bbox[edge[1]][0]
                point.y = single_bbox[edge[1]][1]
                point.z = single_bbox[edge[1]][2]
                marker.points.append(point)

            # Connect the corners to form edges of the box
            marker.lifetime = rospy.Duration(0, self.rviz_marker_retain_duration*10E6)
            marker.pose.orientation.w = 1.0
            marker_list.markers.append(marker)

            if self.debug_enable_visualization:
                # --- Text marker ---
                vis_num += 1
                marker_msg = Marker()
                marker_msg.type = Marker.TEXT_VIEW_FACING
                
                marker_msg.header.stamp = header.stamp
                marker_msg.header.frame_id = self.rviz_cam_output_frame
                
                marker_msg.pose.position.x = cx     # in camera frame: y, left-right
                marker_msg.pose.position.y = cy     # in camera frame: z, height
                marker_msg.pose.position.z = cz     # in camera frame: x, depth         
                marker_msg.pose.orientation.x = 0.0 
                marker_msg.pose.orientation.y = 0.0 
                marker_msg.pose.orientation.z = 0.0
                marker_msg.pose.orientation.w = 1.0 

                marker_msg.scale.z = 0.4 # Size of text
                marker_msg.text = f"{round(distance_of_object, 2)}m. {round(np.rad2deg(-yaw_angle), 2)}deg"
                
                marker_msg.action = Marker.ADD
                
                marker_msg.color.r  = 255
                marker_msg.color.g  = 255
                marker_msg.color.b  = 255
                marker_msg.color.a  = 1.0
                marker_msg.id       = vis_num
                marker_msg.lifetime = rospy.Duration(0, self.rviz_marker_retain_duration*10E6)       
                marker_list.markers.append(marker_msg)
                """
                # --- Corners of BBOX marker ---
                for index, box_points in enumerate(single_bbox):
                    vis_num += 1
                    marker_msg = Marker()
                    marker_msg.type = Marker.TEXT_VIEW_FACING
                    
                    marker_msg.header.stamp = header.stamp
                    marker_msg.header.frame_id = self.rviz_cam_output_frame
                    
                    marker_msg.pose.position.x = box_points[0]     # in camera frame: y, left-right
                    marker_msg.pose.position.y = box_points[1]     # in camera frame: z, height
                    marker_msg.pose.position.z = box_points[2]      # in camera frame: x, depth         
                    marker_msg.pose.orientation.x = 0.0 
                    marker_msg.pose.orientation.y = 0.0 
                    marker_msg.pose.orientation.z = 0.0
                    marker_msg.pose.orientation.w = 1.0 

                    marker_msg.scale.z = 0.4 # Size of text
                    marker_msg.text = f"{index}"
                    
                    marker_msg.action = Marker.ADD  
                    marker_msg.color.r  = 255
                    marker_msg.color.g  = 255
                    marker_msg.color.b  = 255
                    marker_msg.color.a  = 1.0
                    marker_msg.id       = vis_num
                    marker_msg.lifetime = rospy.Duration(0, self.rviz_marker_retain_duration*10E6)             
      
                    marker_list.markers.append(marker_msg)
                
                # --- Axes marker ---
                # NOTE: Used for debugging orientation of boxes for different sensor setup
                # X-Axis: Orange, Y-Axis: Green, Z-Axis: Blue
                for add_point, rgb_color in zip([[2, 0, 0], [0, 2, 0], [0, 0, 2]], [(1.0, 1.0, 0), (0, 1.0, 0), (0, 0, 1.0)]):
                    vis_num += 1
                    marker_msg = Marker()
                    marker_msg.type = Marker.LINE_STRIP
                    
                    marker_msg.header.stamp = header.stamp
                    marker_msg.header.frame_id = self.rviz_cam_output_frame
                         
                    marker_msg.pose.orientation.x = 0.0 
                    marker_msg.pose.orientation.y = 0.0 
                    marker_msg.pose.orientation.z = 0.0
                    marker_msg.pose.orientation.w = 1.0 
                    
                    marker_msg.scale.x = 0.04 # Size of text
                    
                    marker_msg.action = Marker.ADD
                    marker_msg.color.r  = rgb_color[0]
                    marker_msg.color.g  = rgb_color[1]
                    marker_msg.color.b  = rgb_color[2]
                    marker_msg.color.a  = 1.0
                    marker_msg.id       = vis_num
                    marker_msg.lifetime = rospy.Duration(0, self.rviz_marker_retain_duration*10E6)   

                    marker_msg.points = []
                    p1 = Point()
                    p1.x = cx
                    p1.y = cy
                    p1.z = cz

                    p2 = Point()
                    p2.x = cx + add_point[0]
                    p2.y = cy + add_point[1]
                    p2.z = cz + add_point[2]

                    marker_msg.points.append(p1)
                    marker_msg.points.append(p2)

                    marker_list.markers.append(marker_msg)
                    """
        return marker_list

    def transform_img(self, img):
        # Crop off top pixels to remove unnecessary image
        # NOTE: No crop is being used, 3D box coordinates are more incorrect when used
        img = img[self.TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP:img.shape[0], :]
        # Resize Image to target resolution
        img = cv2.resize(img, self.TARGET_RESIZE_RES)
    
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
                    if classID > 2: # Only car and pedestrian class
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

        # Load Model
        cfg = omegaconf.OmegaConf.load(rospy.get_param('~config_path'))
        cfg.MODEL.CKPT = rospy.get_param('~model_checkpoint_path')
        self.dd3d = DD3D(cfg)
    
        # Wait for camera intrinsic data and load it
        cam_info_topic_name = rospy.get_param('~camera_info_topic')
        rospy.loginfo(f"MonoCam3D Obj.Det| Waiting for {cam_info_topic_name} topic")
        cam_info_data = rospy.wait_for_message(cam_info_topic_name, CameraInfo)
        rospy.loginfo(f"MonoCam3D Obj.Det| Success. {cam_info_topic_name} topic received !")
        cam_intrinsic_mtx = np.reshape(np.array(cam_info_data.K), (3,3))
        self.dd3d.load_cam_mtx(cam_intrinsic_mtx)

        # Output publishers
        self.marker_pub = rospy.Publisher(rospy.get_param('~marker_output_topic'), MarkerArray, queue_size=10)
        self.vis_image_pub = rospy.Publisher("rgb_front/vis_image", Image, queue_size=10)
        
        rospy.loginfo("MonoCam3D Obj.Det| Success. Publishers has been created !")
        
        self.cam_intrinsic_mtx = None
        self.cvbridge = CvBridge()

        # Checking for compressed ROS image (used by camera recorded rosbags, not for carla)
        if self.dd3d.realcam_capture:
            sub_image = rospy.Subscriber(rospy.get_param('~camera_img_topic'), CompressedImage, callback=self.callback_image)
            self.ros2cv2 = self.cvbridge.compressed_imgmsg_to_cv2
            self.ros2cv2_arg = "passthrough"
        else:
            sub_image = rospy.Subscriber(rospy.get_param('~camera_img_topic'), Image, callback=self.callback_image)
            self.ros2cv2 = self.cvbridge.imgmsg_to_cv2
            self.ros2cv2_arg = "bgr8"
        rospy.spin()


    def callback_image(self, msg: Image):
        header_info = msg.header
        t_1 = perf_counter()
        cv_image = self.ros2cv2(msg, self.ros2cv2_arg)
        transform_img = self.dd3d.transform_img(cv_image)
        t_2 = perf_counter()
        predictions = self.dd3d.inference_on_single_image(transform_img)
        t_3 = perf_counter()
        marker_list = self.dd3d.output2MarkerArray(predictions, header_info)

        self.marker_pub.publish(marker_list)
        t_4 = perf_counter()

        total_time = (t_4 - t_1)*1000
        if self.dd3d.debug_enable_visualization:
            # NOTE: For debug purposes, not needed anymore
            # final_image = self.dd3d.visualize([cv_image], predictions)[0]
            # ros_img = self.cvbridge.cv2_to_imgmsg(final_image)

            # self.vis_image_pub.publish(ros_img)
            rospy.loginfo(f"MonoCam3D Obj.Det| Num_objects: {len(marker_list.markers)}, Shape: {self.dd3d.TARGET_RESIZE_RES}, Time log| Total: {total_time:.2f}ms, Img_Tf: {(t_2 - t_1)*1000:.2f}ms, Inference: {(t_3 - t_2)*1000:.2f}ms, conv_toMarker: {(t_4 - t_3)*1000:.2f}ms")
        else:
            rospy.loginfo(f"MonoCam3D Obj.Det| Num_objects: {len(marker_list.markers)}, Time per frame: {total_time:.2f}ms, {1000/total_time :.2f}fps")

if __name__ == '__main__':
    random_obj = obj_detection()