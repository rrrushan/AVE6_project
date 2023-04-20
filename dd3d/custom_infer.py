import os
import hydra
import torch

from tqdm import tqdm
from torchinfo import summary
import numpy as np
import cv2
from time import time
import sys

import detectron2.utils.comm as d2_comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluators, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, get_event_storage

import sys
sys.path.append('/home/carla/admt_student/team3_ss23/dd3d')
import tridet.modeling  # pylint: disable=unused-import
import tridet.utils.comm as comm
from tridet.data import build_test_dataloader, build_train_dataloader
from tridet.data.dataset_mappers import get_dataset_mapper
from tridet.data.datasets import random_sample_dataset_dicts, register_datasets
from tridet.evaluators import get_evaluator
from tridet.modeling import build_tta_model
from tridet.utils.s3 import sync_output_dir_s3
from tridet.utils.setup import setup
from tridet.utils.train import get_inference_output_dir, print_test_results
from tridet.utils.visualization import mosaic, save_vis
from tridet.utils.wandb import flatten_dict, log_nested_dict
from tridet.visualizers import get_dataloader_visualizer, get_predictions_visualizer

class DD3D:
    def __init__(self, cfg):
        # Create model architecture
        self.model = build_model(cfg)
        checkpoint_file = cfg.MODEL.CKPT
        
        os.chdir("../../..") # Moving cwd from dd3d/outputs/date/time to dd3d/
        
        # Load trained weights
        if checkpoint_file:
            self.model.load_state_dict(torch.load(checkpoint_file)["model"])
        summary(self.model) # Print model summary

        self.model.eval() # Inference mode

        # Camera Intrinsic Matrix -> Using KiTTi's default values here
        self.cam_intrinsic_mtx = torch.FloatTensor([
            [738.9661,   0.0000, 624.2830],
            [  0.0000, 738.8547, 177.0025],
            [  0.0000,   0.0000,   1.0000]
        ])
        self.cam_intrinsic_mtx = torch.FloatTensor([
            [1440.0,   0.0000, 720.0],
            [  0.0000, 1440.0, 540.0],
            [  0.0000,   0.0000,   1.0000]
        ])

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

    def visualize(self, batched_images, batched_model_output, mode="3D"):
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
            # pred_bbox2d = single_output["instances"].pred_boxes.tensor.detach().cpu().numpy()
            pred_classes = single_output["instances"].pred_classes.detach().cpu().numpy() # ClassIDs of detections
            pred_scores = single_output["instances"].scores_3d.detach().cpu().numpy() # Confidence scores
            pred_bbox3d = single_output["instances"].pred_boxes3d.corners.detach().cpu().numpy() # 3D corners

            ## Project 3D world points to image
            # Multiply the 3D points by the intrinsic matrix to obtain the 2D points
            points_2d_homogeneous = np.dot(pred_bbox3d, self.cam_intrinsic_mtx.cpu().numpy().T) # intrinsic_matrix.dot(points_3d.T).T
            # Convert the homogeneous 2D points to non-homogeneous coordinates
            pred_bbox3d_img = points_2d_homogeneous[:, :, :2] / points_2d_homogeneous[:, :, 2:]

            if mode == "3D":
                # Define the edges of the bounding box.
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ]
                img_alpha = image.copy()

                for box, classID, conf_score in zip(pred_bbox3d_img, pred_classes, pred_scores):
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
        transformed_img = torch.from_numpy(image)
        
        # Transposing: [H, W, C] -> [C, H, W] (KiTTi: [3, 375, 1242])
        transformed_img = transformed_img.permute(2, 0, 1)
    
        output = self.model(transformed_img[None, :], [self.cam_intrinsic_mtx])
        return output
    
@hydra.main(config_path="configs/", config_name="defaults")
def main(cfg):
    dd3d = DD3D(cfg)

    # IMG_PATH = "/home/carla/admt_student/team3_ss23/dd3d/media/input_img_2.png"
    IMG_FOLDER_PATH = "/home/carla/admt_student/team3_ss23/data/KITTI3D/testing/image_2"
    # IMG_FOLDER_PATH = "/home/carla/admt_student/team3_ss23/data/phone_pics"
    total_files = len(os.listdir(IMG_FOLDER_PATH))
    for file_num, file in enumerate(os.listdir(IMG_FOLDER_PATH)):
        print(f"Visualizing file: {file_num}/{total_files}")
        image = cv2.imread(os.path.join(IMG_FOLDER_PATH, file))
        image = cv2.resize(image, (1080, 1440))
        predictions = dd3d.inference_on_single_image(image)
        print(predictions[0]["instances"].pred_classes.detach().cpu().numpy())
        print(predictions[0]["instances"].pred_boxes3d.corners.detach().cpu().numpy())
        exit()
        final_image = dd3d.visualize([image], predictions)[0]
        cv2.imwrite(f"outputs/test_phone/{file}", final_image)

if __name__ == '__main__':
    ## Uncomment for the required model
    # OmniML
    sys.argv.append('+experiments=dd3d_kitti_omninets_custom')
    sys.argv.append('MODEL.CKPT=trained_final_weights/omniml.pth')
    
    # DLA34cam_intrinsic_mtx
    # V99
    # sys.argv.append('+experiments=dd3d_kitti_v99')
    # sys.argv.append('MODEL.CKPT=trained_final_weights/v99.pth')
    
    main()  # pylint: disable=no-value-for-parameter