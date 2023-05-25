#!/usr/bin/env python3

import glob
import os
import sys
import numpy as np
from PIL import Image
import shutil
import random
import cv2 as cv

import carla
import time



import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy

class ScenarioCreator(CompatibleNode):
    def __init__(self):
        super(ScenarioCreator, self).__init__("DistortedImagePublisher")
        self.role_name = self.get_param("role_name", "ego_vehicle")






def main(args=None):
    try: 
        roscomp.init("dist_img_pub", args=args)
        scenario_node = ScenarioCreator()
        scenario_node.loginfo("Scenario node is running")
    finally: 
        roscomp.shutdown()



if __name__ == '__main__':
    main()