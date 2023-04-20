#!/usr/bin/env python
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os
from collections import OrderedDict, defaultdict

import hydra
import torch
import wandb
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from torch.cuda import amp
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from torchinfo import summary
import numpy as np
import cv2
from time import time

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

LOG = logging.getLogger('tridet')


def project_to_image(points_3d, intrinsic_matrix):
    """
    Project 3D points onto a 2D image using an intrinsic matrix.

    Args:
        points_3d (array): A numpy array of shape (N, 3) containing N 3D points in world coordinates.
        intrinsic_matrix (array): A numpy array of shape (3, 3) representing the intrinsic matrix.

    Returns:
        points_2d (array): A numpy array of shape (N, 2) containing the corresponding 2D points in image coordinates.
    """
    # Multiply the 3D points by the intrinsic matrix to obtain the 2D points
    points_2d_homogeneous = np.dot(points_3d, intrinsic_matrix.T) # intrinsic_matrix.dot(points_3d.T).T

    # Convert the homogeneous 2D points to non-homogeneous coordinates
    points_2d = points_2d_homogeneous[:, :, :2] / points_2d_homogeneous[:, :, 2:]
    return points_2d

def draw_boxes_3d(image, boxes_3d, classes, scores):
    """
    Draw 3D boxes on an image using a numpy array of 2D box coordinates.

    Args:
        image (array): A numpy array representing the image.
        boxes_2d (array): A numpy array of shape (N, 8, 2) representing the 2D box coordinates.

    Returns:
        image (array): A numpy array with the 3D boxes drawn on it.
    """
    # BGR (inverted RGB) color values for classes
    color_mapping = {
        0: ((0, 0, 255), "Car"),         # Car: Red
        1: ((0, 255, 0), "Pedestrian"),  # Pedestrian: Green
        2: ((255, 0, 0), "Cyclist"),     # Cyclist: Blue
        3: ((0, 192, 255), "Van"),       # Van: Yellow
        4: ((255, 255, 0), "Truck"),     # Truck: Turquoise Blue
        5: ((0, 0, 0), "Unknown")        # Unknown: Black
    }
    # Define the edges of the bounding box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    img_alpha = image.copy()

    for box, classID, conf_score in zip(boxes_3d, classes, scores):
        class_color = color_mapping[int(classID)][0]
        class_name = color_mapping[int(classID)][1]
        # Draw the edges of the bounding box
        for edge in edges:
            start_point = tuple(box[edge[0]].astype(int))
            end_point = tuple(box[edge[1]].astype(int))
            cv2.line(image, start_point, end_point, class_color, 2, cv2.LINE_AA)
        
        # Plot class name and score
        baseLabel = f'{class_name} {round(conf_score*100)}%'
        (w1, h1), _ = cv2.getTextSize(
            baseLabel, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1
        )
        
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
            
        
        #2D
        # print(box, box.shape)

        # print(np.amin(box, 0), np.amin(box, 1))
        # min_x, min_y = np.amin(box, 0)
        # max_x, max_y = np.amax(box, 0)
        # first_point = (int(min_x), int(min_y))
        # second_point = (int(max_x), int(max_y))

        # cv2.rectangle(
        #     image, 
        #     first_point, 
        #     second_point, 
        #     color_mapping[int(classID)]
        # )

        # DEBUG get points order
        # for index, point in enumerate(box):
        #     point_xy = (int(point[0]), int(point[1]))
        #     text = f"{index}"
        #     cv2.putText(image, text, point_xy, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255))
             
        # Fill just the front face
        points = np.array([
            tuple(box[0].astype(int)), tuple(box[1].astype(int)), 
            tuple(box[2].astype(int)), tuple(box[3].astype(int))
        ])
        
        # Use fillPoly() function and give input as image,
        cv2.fillPoly(img_alpha, pts=[points], color=class_color)

    # # To create transparency effect of the shaded box
    blend = cv2.addWeighted(image, 0.75, img_alpha, 0.25, 0)

    return blend
    


def benchmark(model, ori_image, img_sizes, batchsizes):
    for _img_size in img_sizes:
        image = cv2.resize(ori_image, _img_size)
        # Convert numpy image format to tensor image format 
        # Shape: [H, W, C] -> [B, C, H, W]
        transformed_img = torch.from_numpy(image)
        
        # Transposing
        transformed_img = transformed_img.permute(2, 0, 1)
        # Loading KiTTi3D intrinsics -> Extracted from eval.py
        intrinsic_mtx = torch.FloatTensor([
            [738.9661,   0.0000, 624.2830],
            [  0.0000, 738.8547, 177.0025],
            [  0.0000,   0.0000,   1.0000]
        ])

        input_dict = {}
        input_dict["image"] = transformed_img
        input_dict["intrinsics"] = intrinsic_mtx
        
        for bs in batchsizes:
            time_avg = []
            for loop_num in range(100):
                start_time = time()
                output = model([input_dict]*bs)
                end_time = time()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if loop_num > 0:
                    time_avg.append(end_time - start_time)
            print(f"Img_size: {_img_size}, Batchsize: {bs}, avg time required: {np.mean(time_avg)}")

# @hydra.main(config_path="../configs/", config_name="defaults")
@hydra.main(config_path="../configs/", config_name="defaults")
def main(cfg):
    
    # dataset_names = register_datasets(cfg)
    # if cfg.ONLY_REGISTER_DATASETS:
    #     return {}, cfg
    # LOG.info(f"Registered {len(dataset_names)} datasets:" + '\n\t' + '\n\t'.join(dataset_names))
    
    model = build_model(cfg)

    checkpoint_file = cfg.MODEL.CKPT
    print(cfg.MODEL.CKPT)
    if checkpoint_file:
        # Checkpointer(model).load(checkpoint_file)
        print(torch.load(checkpoint_file).keys())
        model.load_state_dict(torch.load(checkpoint_file)["model"])
    summary(model)
    
    model.eval()
    intrinsic_mtx = torch.FloatTensor([
        [738.9661,   0.0000, 624.2830],
        [  0.0000, 738.8547, 177.0025],
        [  0.0000,   0.0000,   1.0000]
    ])


    # IMG_PATH = "/home/carla/admt_student/team3_ss23/dd3d/media/input_img_2.png"
    IMG_FOLDER_PATH = "/home/carla/admt_student/team3_ss23/data/KITTI3D/testing/image_2"
    total_files = len(os.listdir(IMG_FOLDER_PATH))
    for file_num, file in enumerate(os.listdir(IMG_FOLDER_PATH)):
        print(f"Visualizing file: {file_num}/{total_files}")
        image = cv2.imread(os.path.join(IMG_FOLDER_PATH, file))
        # image = cv2.resize(image, (1920, 1080))
        # Convert numpy image format to tensor image format 
        # Shape: [H, W, C] -> [B, C, H, W]
        # transformed_img = torch.from_numpy(image[:, :, [2, 1, 0]])
        transformed_img = torch.from_numpy(image)
        # Adding batch-dimension and then transposing
        # transformed_img = transformed_img[np.newaxis, :] 
        # transformed_img = transformed_img.permute(0, 3, 1, 2)
        
        # Transposing
        transformed_img = transformed_img.permute(2, 0, 1)
        # Loading KiTTi3D intrinsics -> Extracted from eval.py
        
        input_dict = {}
        input_dict["image"] = transformed_img
        input_dict["intrinsics"] = intrinsic_mtx
    
        output = model([input_dict])

        pred_bbox2d = output[0]["instances"].pred_boxes.tensor.detach().cpu().numpy()
        pred_bbox3d = output[0]["instances"].pred_boxes3d.corners.detach().cpu().numpy() # 3D corners
        pred_bbox3d_img = project_to_image(pred_bbox3d, intrinsic_mtx.cpu().numpy())

        pred_classes = output[0]["instances"].pred_classes.detach().cpu().numpy()
        pred_scores = output[0]["instances"].scores_3d.detach().cpu().numpy()
        # benchmark(model, image, 
        #     # [(512, 512), (640, 480), (960, 540), (1280, 720), (1600, 900), (1920, 1080)], 
        #     # [1, 2]
        #     [(1600, 900), (1920, 1080)], 
        #     [1]
        # )
        # visualize
        color_mapping = {
            0: (255, 0, 0),
            1: (0, 255, 0),
            2: (0, 0, 255),
            3: (255, 255, 0),
            4: (0, 255, 255),
            5: (0, 0, 0)
        }
        # for bbox2d, classid in zip(pred_bbox2d, pred_classes):
        #     cv2.rectangle(
        #         image, 
        #         (int(bbox2d[0]), int(bbox2d[1])), 
        #         (int(bbox2d[2]), int(bbox2d[3])), 
        #         color_mapping[int(classid)])
        output_img = draw_boxes_3d(image, pred_bbox3d_img, pred_classes, pred_scores)
        # cv2.imwrite("/home/carla/admt_student/team3_ss23/dd3d/outputs/vis_output_omniml_3d_2.png", output_img)
        cv2.imwrite(f"/home/carla/admt_student/team3_ss23/dd3d/outputs/testing_output/{file}", output_img)
    exit()

if __name__ == '__main__':
    sys.argv.append('+experiments=dd3d_kitti_omninets_custom')
    sys.argv.append('MODEL.CKPT=/home/carla/admt_student/team3_ss23/dd3d/outputs/2023-04-05/13-28-08/model_final.pth')
    # sys.argv.append('+experiments=dd3d_kitti_dla34')
    # sys.argv.append('MODEL.CKPT=/home/carla/admt_student/team3_ss23/dla34_exp.pth')
    # sys.argv.append('+experiments=dd3d_kitti_v99')
    # sys.argv.append('MODEL.CKPT=/home/carla/admt_student/team3_ss23/v99_exp.pth')
    
    sys.argv.append('EVAL_ONLY=True')
    sys.argv.append('TEST.IMS_PER_BATCH=4')
    
    
    main()  # pylint: disable=no-value-for-parameter
    LOG.info("DONE.")