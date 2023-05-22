#!/usr/bin/env python3

import glob
import os
import sys
import numpy as np
from PIL import Image
import shutil
import random
import cv2 as cv

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
    # client = carla.Client('10.116.80.2', 2000)
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    for i in world.get_actors():
        if i.type_id == 'vehicle.tesla.model3':
            world.get_actor(i.id).destroy()

    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')

    color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
    if vehicle_bp.has_attribute('color'):
        vehicle_bp.set_attribute('color', color)

    vehicle_transform = carla.Transform(carla.Location(-47, -40, 0.6), carla.Rotation(yaw=90))
    vehicle_tesla = world.spawn_actor(vehicle_bp, vehicle_transform)
    time.sleep(5)

    blueprintsWalkers = world.get_blueprint_library().filter("walker.pedestrian.*")
    walker_bp = random.choice(blueprintsWalkers)
    print(walker_bp)
    walker_transform = carla.Transform(carla.Location(12.0, 0, 1), carla.Rotation(yaw=90))
    walker = world.spawn_actor(walker_bp,walker_transform,attach_to=vehicle_tesla, attachment_type=carla.AttachmentType.Rigid)
    
    cam_bp = None
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute("image_size_x",str(1936))
    cam_bp.set_attribute("image_size_y",str(1464))
    cam_bp.set_attribute("fov",str(60))


    cam_location = carla.Location(x=2, y=0, z=2)
    cam_rotation = carla.Rotation(roll=0, pitch=0, yaw=0) # pitch, yaw, roll
    cam_transform = carla.Transform(cam_location,cam_rotation)
        
    ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=vehicle_tesla, attachment_type=carla.AttachmentType.Rigid)

    ego_cam.listen(lambda image: image.save_to_disk(file_path + '/%.6d.png' % image.frame))
    time.sleep(1.4)
    ego_cam.stop()

    ego_cam.destroy()
    walker.destroy()
    vehicle_tesla.destroy()


if __name__ == '__main__':

    main()