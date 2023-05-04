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
from omegaconf import OmegaConf

@hydra.main(config_path="configs/", config_name="defaults")
def main(cfg):
    yaml_data = OmegaConf.create(cfg)
    with open("/home/carla/admt_student/team3_ss23/AVE6_project/dd3d/outputs/omniml.yaml", "w") as fp:
        OmegaConf.save(config=yaml_data, f=fp.name)

if __name__ == '__main__':
    ## Uncomment for the required model
    # OmniML
    sys.argv.append('+experiments=dd3d_kitti_omninets_custom')
    sys.argv.append('MODEL.CKPT=trained_final_weights/omniml.pth')
    
    # DLA34
    # sys.argv.append('+experiments=dd3d_kitti_dla34')
    # sys.argv.append('MODEL.CKPT=trained_final_weights/dla34.pth')

    # V99
    # sys.argv.append('+experiments=dd3d_kitti_v99')
    # sys.argv.append('MODEL.CKPT=trained_final_weights/v99.pth')

    # DLA34 Nuscenes
    # sys.argv.append('+experiments=dd3d_nusc_dla34_custom')
    # sys.argv.append('MODEL.CKPT=trained_final_weights/dla34_nusc_step6k.pth')
    main()  # pylint: disable=no-value-for-parameter