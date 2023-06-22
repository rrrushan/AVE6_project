#!/usr/bin/env python3

'''
This script finds the sensors attached to the vehicle in CARLA ROS bridge and displays
their coordinates within the map. The script was used to confirm that the sensors were 
placed in the correct coordinates, corresponding to coordinates of the real sensors in
the real tests.
'''

import glob
import os
import sys
import random
import numpy as np


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
    client = carla.Client('10.116.80.2', 2000)
    # client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # spawn a temporary vehicle to enable actor list
    vehicle_bp = world.get_blueprint_library().find('vehicle.diamondback.century')  #vehicle.tesla.model3
    vehicle_transform = carla.Transform(carla.Location(-67, -40, 0.6), carla.Rotation(yaw=90))
    vehicle_tesla = world.spawn_actor(vehicle_bp, vehicle_transform)
    time.sleep(1)

    # destroy previous targets
    actor_list = world.get_actors()

    for object in actor_list.filter('sensor.camera.rgb'):
        print(object.get_transform())

   
if __name__ == '__main__':
    main()