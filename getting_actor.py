#!/usr/bin/env python3


import glob
import os
import sys
import numpy as np
from PIL import Image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
import time

def main():

    try:

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
                
        world = client.get_world()

        flag = False
        if flag:
            actors = world.get_actors()
            # print(actors.get_location())
            # print(actors.has_attribute(84))
            for i in range(len(actors)):
                print(actors[i], i)
            
        else:
        
            # ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
            vehicle = world.get_actor(42)
            cam_bp = None
            cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            cam_bp.set_attribute("image_size_x",str(1920))
            cam_bp.set_attribute("image_size_y",str(1080))
            cam_bp.set_attribute("fov",str(80))
            #cam_bp.set_attribute("lens_circle_falloff",str())
            #cam_bp.set_attribute("lens_circle_multiplier",str(5))
            #cam_bp.set_attribute("lens_k",str(-5))
            #cam_bp.set_attribute("lens_kcube",str(-5))
            #cam_bp.set_attribute("lens_x_size",str(6))
            #cam_bp.set_attribute("lens_y_size",str(6))
            cam_location = carla.Location(2.356953, 0.000000, 1.618680)
            cam_rotation = carla.Rotation(0.000000 ,0.000000 ,0.000000)
            cam_transform = carla.Transform(cam_location,cam_rotation)
            
            ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
    
            # ego_cam.listen(lambda image: image.save_to_disk('./output/%.6d.jpg' % image.frame))
            
            time.sleep(20)



    finally:
        pass

main()