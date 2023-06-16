import os
import torch
from tqdm import tqdm
import numpy as np
import cv2
from time import time, perf_counter

print("--- DD3D Inference on single images ---")
import omegaconf
import detectron2.utils.comm as d2_comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluators, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, get_event_storage

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

# --- Global variables (defined for ROS in .roslaunch files) ---
PRE_NMS_THRESH = 0.1
NMS_THRESH = 0.2
TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP = 0

MIN_TRACKWIDTH = 1.3
MAX_TRACKWIDTH = 2.5
MIN_WHEELBASE = 3.5
MAX_WHEELBASE = 5.8
MIN_HEIGHT = 0.5
MAX_HEIGHT = 2.5

PED_MAX_BASE = 2.5
PED_MIN_HEIGHT = 1.2
PED_MAX_HEIGHT = 2.2
# ---

class DD3D:
    def __init__(self, cfg, input_img_res, res):
        # Create model 
        # TODO: Convert to param Caro
        cfg.DD3D.FCOS2D.INFERENCE.PRE_NMS_THRESH = PRE_NMS_THRESH # 0.1
        cfg.DD3D.FCOS2D.INFERENCE.NMS_THRESH = NMS_THRESH  # 0.2 
        # --
        try:
            self.model = build_model(cfg)
        except:
            print("ERRO in loading model .cfg. Check path !!")
            exit()
        print("Model config .cfg loaded successfully")
        checkpoint_file = cfg.MODEL.CKPT
        
        os.chdir("../../..") # Moving cwd from dd3d/outputs/date/time to dd3d/
        
        # Load trained weights
        try: 
            if checkpoint_file:
                self.model.load_state_dict(torch.load(checkpoint_file)["model"])
            # summary(self.model) # Print model summary
        except:
            print("ERROR in loading model checkpoint .pth. Check path !!")
            exit()
        print("Model checkpoint  pth loaded successfully")
        self.model.eval() # Inference mode

        print(input_img_res)
        # Params for cropping and rescaling
        self.ORIG_IMG_HEIGHT = input_img_res[1]  # 1464 
        self.ORIG_IMG_WIDTH = input_img_res[0]    # 1936 
        self.TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP = TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP
        self.CROPPED_IMG_HEIGHT = self.ORIG_IMG_HEIGHT - self.TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP
        target_resize_width = res
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

        self.car_min_trackwidth = MIN_TRACKWIDTH
        self.car_max_trackwidth = MAX_TRACKWIDTH
        self.car_min_wheelbase  = MIN_WHEELBASE
        self.car_max_wheelbase  = MAX_WHEELBASE
        self.car_min_height     = MIN_HEIGHT
        self.car_max_height     = MAX_HEIGHT

        self.ped_max_base   = PED_MAX_BASE
        self.ped_min_height = PED_MIN_HEIGHT
        self.ped_max_height = PED_MAX_HEIGHT
        
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

    # NOTE: Commented out since only relevant for ROS
    # def output2MarkerArray(self, predictions, header):
    #     """Converts model outputs to marker array format for RVIZ

    #     Args:
    #         predictions (list[Instance]): 
    #             Each item in the list has type Instance. Has all detected data
    #     Returns:
    #         marker_list (list): List of Markers for RVIZ
    #     """

    #     def get_quaternion_from_euler(roll, pitch, yaw):
    #         """
    #         Convert an Euler angle to a quaternion.

    #         Input
    #             :param roll: The roll (rotation around x-axis) angle in radians.
    #             :param pitch: The pitch (rotation around y-axis) angle in radians.
    #             :param yaw: The yaw (rotation around z-axis) angle in radians.

    #         Output
    #             :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    #         """
    #         qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    #         qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    #         qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    #         qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            
    #         return qx, qy, qz, qw

    #     bboxes_per_image = predictions[0]["instances"].pred_boxes3d.corners.detach().cpu().numpy()
    #     classes_per_image = predictions[0]["instances"].pred_classes.detach().cpu().numpy()
    #     marker_list = MarkerArray() # []
        
    #     vis_num = 0 
    #     for index, (single_bbox, single_class) in enumerate(zip(bboxes_per_image, classes_per_image)):
    #         class_id = int(single_class)
    #         if class_id > 1: # Car class: 0, Pedestrian class: 1. Ignoring all the other classes
    #             continue

    #         vis_num += 1

    #         marker_msg = Marker()
    #         # BBOX is stored as (8, 3) -> where last_index is (y, z, x)
    #         min_x, max_x = np.min(single_bbox[:, 0]), np.max(single_bbox[:, 0])
    #         min_y, max_y = np.min(single_bbox[:, 1]), np.max(single_bbox[:, 1])
    #         min_z, max_z = np.min(single_bbox[:, 2]), np.max(single_bbox[:, 2])

    #         # Center points of BBOX along each axis
    #         cx = (min_x + max_x) / 2
    #         cy = (min_y + max_y) / 2
    #         cz = (min_z + max_z) / 2
            
    #         if cz <= 1.5:
    #             continue

    #         # Limiting size of BBOX in each axis according parameters in ROS Launch
    #         if class_id == 0: # Car
    #             scale_x = np.clip(abs(single_bbox[0, 0] - single_bbox[1, 0]), self.car_min_trackwidth, self.car_max_trackwidth) 
    #             scale_y = np.clip(max_y - min_y, self.car_min_height, self.car_max_height)
    #             scale_z = np.clip(max_z - min_z, self.car_min_wheelbase, self.car_max_wheelbase)

    #         elif class_id == 1: # Pedestrian
    #             scale_x = np.clip(abs(max_x - min_x), 0.0, self.ped_max_base) 
    #             scale_y = np.clip(max_y - min_y, self.ped_min_height, self.ped_max_height)
    #             scale_z = np.clip(max_z - min_z, 0.0, self.ped_max_base)
            

    #         ## Rotation of BBOX along each axis
    #         # Setting Roll and Pitch angle to 0.0 as they the vehicles are considered to be on flat surface.
    #         # To further reduce noisy bbox positions/scales
    #         pitch_angle = 0.0 # np.math.atan2((single_bbox[4, 1] - single_bbox[0, 1]), (single_bbox[4, 2] - single_bbox[0, 2]),) # pitch in camera
    #         roll_angle = 0.0 # np.math.atan2((single_bbox[1, 1] - single_bbox[0, 1]), (single_bbox[1, 0] - single_bbox[0, 0]))
    #         yaw_angle = -np.math.atan2((single_bbox[1, 2] - single_bbox[0, 2]), (single_bbox[1, 0] - single_bbox[0, 0])) # yaw in camera.
            
    #         qx, qy, qz, qw = get_quaternion_from_euler(pitch_angle, yaw_angle, roll_angle)
            
    #         marker_msg.type = Marker.CUBE
    #         marker_msg.header.stamp = header.stamp
    #         marker_msg.header.frame_id = "ego_vehicle/rgb_front"
            
    #         marker_msg.pose.position.x = cx     # in camera frame: y, left-right
    #         marker_msg.pose.position.y = cy     # in camera frame: z, height
    #         marker_msg.pose.position.z = cz     # in camera frame: x, depth 
    #         marker_msg.pose.orientation.x = qx 
    #         marker_msg.pose.orientation.y = qy 
    #         marker_msg.pose.orientation.z = qz 
    #         marker_msg.pose.orientation.w = qw 
            
            
    #         marker_msg.scale.x = scale_x # Trackwidth
    #         marker_msg.scale.y = scale_y # Height
    #         marker_msg.scale.z = scale_z # Wheelbase
            
    #         """
    #         # first angle -> pitch, yaw, roll
    #         qx, qy, qz, qw = get_quaternion_from_euler(np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(10.0))
    #         marker_msg.pose.position.x = 3.0     # in camera frame: y, left-right
    #         marker_msg.pose.position.y = 1.0     # in camera frame: z. height
    #         marker_msg.pose.position.z = 10.0    # in camera frame: x, depth
    #         marker_msg.pose.orientation.x = qx #- 0.5
    #         marker_msg.pose.orientation.y = qy #+ 0.5
    #         marker_msg.pose.orientation.z = qz #- 0.5
    #         marker_msg.pose.orientation.w = qw #+ 0.5
            
    #         marker_msg.scale.x = 2.0 # scale_x, trackwidth
    #         marker_msg.scale.y = 1.0 # scale_y. height of the car
    #         marker_msg.scale.z = 4.0 # scale_z, wheelbase
    #         """

    #         color               = self.color_mapping[class_id][0]
    #         marker_msg.color.r  = color[2]
    #         marker_msg.color.g  = color[1]
    #         marker_msg.color.b  = color[0]
    #         marker_msg.color.a  = 0.5
    #         marker_msg.id       = vis_num
    #         marker_msg.lifetime = rospy.Duration(0, 3 * 10E7)
            
    #         marker_list.markers.append(marker_msg)

    #         marker_msg = Marker()
            
    #     return marker_list

    def transform_img(self, img):
        img = img[self.TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP:img.shape[0], :]
        img = cv2.resize(img, self.TARGET_RESIZE_RES)
       
        return img

    def rescale_boxes(self, boxes):
        """Rescale boxes after cropping and padding

        Args:
            boxes (np.array): 3D boxes. Shape: [Batchsize, 8, 2]

        Returns:
            final_box_points (np.array): Modified 3D boxes. Shape: [Batchsize, 8, 2]
        """
       
        remove_resize_x, remove_resize_y = boxes[:, :, 0] / self.aspect_ratio, boxes[:, :, 1] / self.aspect_ratio 
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
                    if classID > 2: # Only car, pedestrian, cyclist class
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

        #with torch.cuda.amp.autocast():
            
        output = self.model.predict(transformed_img[None, :], [self.cam_intrinsic_mtx])

        return output

def bench(cfg_path, ckpt_path):
    cfg = omegaconf.OmegaConf.load(cfg_path)

    #cfg.MODEL.CKPT = "/home/carla/admt_student/team3_ss23/AVE6_project/dd3d/trained_final_weights/v99.pth" # v99.pth
    cfg.MODEL.CKPT = ckpt_path
    image = cv2.imread("/home/carla/admt_student/team3_ss23/ROS_1/bag_imgs/selected_imgs/738.png")
    cam_mtx = np.array([1676.6251817266734, 0.0, 968.0, 0.0, 1676.6251817266734, 732.0, 0.0, 0.0, 1.0]).reshape(3, 3)    
    
    resolutions = [968, 640, 484]
    
    NUM_ITERS = 50
    output_strings = []
    for res in resolutions:
        dd3d = DD3D(cfg, res)
        dd3d.load_cam_mtx(cam_mtx)
        
        time_transform = []
        time_inference = []
        for index in tqdm(range(NUM_ITERS)):
            t1 = perf_counter()
            transformed_img = dd3d.transform_img(image)
            t2 = perf_counter()
            predictions = dd3d.inference_on_single_image(transformed_img)
            t3 = perf_counter()

            if index > 5:
                time_transform.append(t2 - t1)
                time_inference.append(t3 - t2)
        
        output_strings.append(f"MODEL: v2_99, Resolution: {dd3d.TARGET_RESIZE_RES}, transform time: {np.mean(time_transform)*1000:.2f}, infer time: {np.mean(time_inference) * 1000:.2f}ms")
        final_img = dd3d.visualize([transformed_img], predictions, "3D", False)[0]
        cv2.imwrite(f"/home/carla/admt_student/team3_ss23/AVE6_project/dd3d/outputs/new_test/738_infer_v99_fp32_{res}.png", final_img)
        del dd3d
    
    for out_str in output_strings:
        print(out_str)
    
def infer_on_img_folder(cfg_path, ckpt_path, img_folder, output_folder, cam_mtx, input_img_res, output_res=1452):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"\n\nSaving output images in folder: {output_folder}")
    
    cfg = omegaconf.OmegaConf.load(cfg_path)
    cfg.MODEL.CKPT = ckpt_path
    
    dd3d = DD3D(cfg, input_img_res, output_res)
    dd3d.load_cam_mtx(cam_mtx)
    
    for file_path in tqdm(os.listdir(img_folder)):
        if file_path.split(".")[-1] in ["png", "jpg"]:
            image = cv2.imread(os.path.join(img_folder, file_path))

            transformed_img = dd3d.transform_img(image)
            predictions = dd3d.inference_on_single_image(transformed_img)
            final_img = dd3d.visualize([image], predictions, "3D", True)[0]
            
            cv2.imwrite(os.path.join(output_folder, file_path), final_img)

    print("--- Finished visualizing images ---")
        
if __name__ == '__main__':
    # --- Change the below 6 variables ---
    orig_img_height = 1464
    orig_img_width = 1936
    cam_mtx = np.array([1676.6251817266734, 0.0, 968.0, 0.0, 1676.6251817266734, 732.0, 0.0, 0.0, 1.0]).reshape(3, 3)    
    # NOTE: If you dont know camera matrix, use the below code to approximate. Maynot be accurate
    # cam_mtx = np.array([orig_img_width, 0.0, orig_img_width/2, 0.0, orig_img_width, orig_img_height/2, 0.0, 0.0, 1.0]).reshape(3, 3)    
    
    output_img_res = 1280 # Use less than 1600. Too big res might not fit in GPU
    model = "v99" # v99 or dla
    img_folder_path = "/home/carla/admt_student/team3_ss23/ROS_1/bag_imgs/selected_imgs"
    # ---
    
    cfg_path = f"./dd3d/trained_final_weights/{model}.yaml"
    checkpoint_path = os.path.abspath(f"./dd3d/trained_final_weights/{model}.pth")
    output_folder_path = os.path.abspath(f"./dd3d/outputs/{model}_im{output_img_res}_{img_folder_path.split('/')[-1]}")
    
    infer_on_img_folder(
        cfg_path, checkpoint_path, 
        img_folder_path, output_folder_path, 
        cam_mtx, (orig_img_width, orig_img_height), output_img_res
    )