#!/usr/bin/env python3


import glob
import os
import sys
import numpy as np
from PIL import Image
import shutil
import random
import cv2 as cv


# locations = np.load('/home/carla/AVE6_project/calibration/Carla/locations.npy', allow_pickle=True) 
# rotations = np.load('/home/carla/AVE6_project/calibration/Carla/rotations.npy', allow_pickle=True) 
        
# print(locations, rotations)
# exit()

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

# def cam_callback(img):
#     cv.imshow("test", img)
#     cv.waitKey(0)


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

        # locations = np.load('/home/carla/AVE6_project/calibration/Carla/locations.npy', allow_pickle=True) 
        # rotations = np.load('/home/carla/AVE6_project/calibration/Carla/rotations.npy', allow_pickle=True)

        locations = np.array([[2.0, -1.8, -1.1], [2.0, -1.6, -1.1]])
        rotations = np.array([[0.0, 20.0, 30.0], [0.0, 20.0, 30.0]])
        
        # Rotation value, position of the camera to center(x, y, z) checkerboard for the corresponding rotation 
        rot_loc_list = [
            ([0.0, -20.0, 0.0], [2.0, 0.0, 0.85]),
            ([0.0, -30.0, 0.0], [2.0, 0.0, 1.5]),
            ([0.0, 0.0, 20.0], [2.0, -1.0, 0.0]),
            ([0.0, 0.0, 30.0], [2.0, -1.6, 0.0]),
            ([0.0, 5.0, 10.0], [2.0, -0.85, -0.2]),
            ([0.0, 20.0, 30.0], [2.0, -1.8, -1.1]),
        ]
        y_lim = (-1.0, 1.0)
        z_lim = (-0.85, 0.85)
        number_of_steps = 4
        rot_x_lim = 10

        locations = []
        rotations = []

        for item in rot_loc_list:
            loc = item[1]
            rot = item[0]

            for y_offset in np.linspace(y_lim[0], y_lim[1], number_of_steps):
                for z_offset in np.linspace(z_lim[0], z_lim[1], number_of_steps):
                    locations.append([loc[0], loc[1]+y_offset, loc[2]+z_offset])
                    rotations.append([np.random.uniform(-rot_x_lim, rot_x_lim), rot[1], rot[2]])
        print(len(locations))

        for i in world.get_actors():
            if i.type_id == 'vehicle.tesla.model3':
                world.get_actor(i.id).destroy()

        vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')

        color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', color)

        vehicle_transform = carla.Transform(carla.Location(-47.4, 13, 0.6), carla.Rotation(yaw=90))
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

            cam_location = carla.Location(x=loc[0]+2.5, y=loc[1], z=loc[2]+3.4)
            cam_rotation = carla.Rotation(roll=R[0], pitch=R[1], yaw=R[2]) # pitch, yaw, roll
            cam_transform = carla.Transform(cam_location,cam_rotation)
            
            ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=vehicle_tesla, attachment_type=carla.AttachmentType.Rigid)
       
            ego_cam.listen(lambda image: image.save_to_disk(file_path + '/%.6d.png' % image.frame))
            # ego_cam.listen(lambda image: cam_callback(image.frame))
            

            # time.sleep(0.2)
            time.sleep(1.1)
            ego_cam.stop()

            ego_cam.destroy()

        vehicle_tesla.destroy()


if __name__ == '__main__':

    main()