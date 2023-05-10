#!/usr/bin/env python3


import glob
import os
import sys
import numpy as np
from PIL import Image
import shutil
import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time


file_path = './output'
    
if os.path.exists(file_path):
    shutil.rmtree(file_path)


def main():
    client = carla.Client('10.116.80.2', 2000)
    # client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    # print(world)
    
    # actors1 = world.get_actor(61)
    # actors2 = world.get_actor(62)
    # actors3 = world.get_actor(63)

    # actors1.destroy()
    # actors2.destroy()
    # actors3.destroy()

    flag = False
    if flag:
        print("True")
        actors = world.get_actor(47)
        # print(actors)
        print(actors.get_transform().location)
        # print(actors.has_attribute(84))
        # for i in range(len(actors)):
        #     print(actors[i], i)
        
    else:
    
        # # ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        # vehicle = world.get_actor(47)

        locations = np.load('/home/carla/AVE6_project/calibration/Carla/locations_2.npy', allow_pickle=True) 
        rotations = np.load('/home/carla/AVE6_project/calibration/Carla/rotations_2.npy', allow_pickle=True) 

        for i in world.get_actors():
            if i.type_id == 'vehicle.tesla.model3':
                world.get_actor(i.id).destroy()

        vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')

        color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', color)

        vehicle_transform = carla.Transform(carla.Location(-47.5, 5, 0.6), carla.Rotation(yaw=90))
        vehicle_tesla = world.spawn_actor(vehicle_bp,vehicle_transform)
        time.sleep(5)

        cam_bp = None
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(1936))
        cam_bp.set_attribute("image_size_y",str(1464))
        cam_bp.set_attribute("fov",str(60))

        # cam_bp.set_attribute("lens_circle_falloff",str(3.0))
        # cam_bp.set_attribute("lens_circle_multiplier",str(3.0))
        #cam_bp.set_attribute("lens_k",str(-5))
        #cam_bp.set_attribute("lens_kcube",str(-5))c
        #cam_bp.set_attribute("lens_x_size",str(6))
        #cam_bp.set_attribute("lens_y_size",str(6))
        
        for i in range(len(locations)):
            loc = locations[i]
            R = rotations[i]

            cam_location = carla.Location(x=loc[1]+2.8, y=loc[0], z=loc[2]+2.5)
            cam_rotation = carla.Rotation(roll=R[1], pitch=R[0], yaw=R[2]) # pitch, yaw, roll
            cam_transform = carla.Transform(cam_location,cam_rotation)
            
            ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=vehicle_tesla, attachment_type=carla.AttachmentType.Rigid)
       
            ego_cam.listen(lambda image: image.save_to_disk(file_path + '/%.6d.png' % image.frame))
            # time.sleep(0.2)
            time.sleep(1)
            ego_cam.stop()

            ego_cam.destroy()

        vehicle_tesla.destroy()


if __name__ == '__main__':

    main()