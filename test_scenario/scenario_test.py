#!/usr/bin/env python3

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

    # debug = world.debug
    # spawn_points = world.get_map().get_spawn_points()
    # print(len(spawn_points))
    # for sp in spawn_points:
    #     debug.draw_point(sp.location, size=0.1, color = carla.Color(0, 255, 0), life_time=-1.0)
    #     # print(sp.location)
    #     # print(sp.rotation)
    #     # print('__')

    # exit()

    # scenario flags
    rain = False

    # target_type = "pedestrian"
    target_type = "bicycle"
    # target_type = "car"

    camera_pos_x = 2
    distance = 5 + camera_pos_x # debug value
    # distance = 28 + camera_pos_x
    # distance = 56 + camera_pos_x
    # distance = 84 + camera_pos_x
    # distance = 112 + camera_pos_x


    # weather settings
    weather = world.get_weather()
    # weather.cloudiness=20.000000
    # weather.precipitation=0.000000
    # weather.precipitation_deposits=0.000000
    # weather.wind_intensity=0.000000
    # weather.sun_azimuth_angle=360.000000
    # weather.sun_altitude_angle=90.000000
    # weather.fog_density=2.000000
    # weather.fog_distance=0.750000
    # weather.fog_falloff=0.100000
    # weather.wetness=0.000000
    # weather.scattering_intensity=1.000000
    # weather.mie_scattering_scale=0.030000
    # weather.rayleigh_scattering_scale=0.033100
    if rain:
        weather.precipitation=100.000000
    world.set_weather(weather)


    # spawn a vehicle to enable actor list
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.cybertruck')  #vehicle.tesla.model3
    vehicle_transform = carla.Transform(carla.Location(-67, -40, 0.6), carla.Rotation(yaw=90))
    vehicle_tesla = world.spawn_actor(vehicle_bp, vehicle_transform)
    time.sleep(1)


    # destroy previous targets
    actor_list = world.get_actors()
    for walker in actor_list.filter('walker.*'):
        walker.destroy()
    for vehicle in actor_list.filter('vehicle.*'):
        if vehicle.type_id != 'vehicle.tesla.model3':
            vehicle.destroy()

    # get the ros bridge vehicle
    for vehicle in actor_list.filter('vehicle.*'):
        if vehicle.type_id == 'vehicle.tesla.model3':
            ego_vehicle = vehicle
            ego_vehicle_transform = vehicle.get_transform()
            print("ego vehicle transform: ", ego_vehicle_transform)

    # vehicle_tesla.destroy() # destroyed with other vehicles above

    # temporary targets
    if target_type == "pedestrian": 
        pass
        # blueprintsTarget = world.get_blueprint_library().filter("walker.pedestrian.*")
        # target_bp = random.choice(blueprintsTarget)
        # target_transform = carla.Transform(carla.Location(distance, 0, 1), carla.Rotation(yaw=90))
        # world.spawn_actor(target_bp,target_transform, attach_to=ego_vehicle, attachment_type=carla.AttachmentType.Rigid)
    elif target_type == "bicycle":
        # load bike
        pass
    elif target_type == "car":
        # load car
        pass
    else:
        print("Wrong target type")
        exit()

    # finding coordiantes of the target on a straight line in front of the vehicle
    vehicle_yaw = ego_vehicle_transform.rotation.yaw * np.pi / 180
    target_vec = np.array([distance, 0, 0])
    rotation_mat = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0], 
                             [np.sin(vehicle_yaw),  np.cos(vehicle_yaw), 0],
                             [0                  ,                    0, 1]])
    target_vec_rot = rotation_mat @ target_vec.T

    vehicle_location = ego_vehicle_transform.location

    target_x = vehicle_location.x + target_vec_rot[0]
    target_y = vehicle_location.y + target_vec_rot[1]
    target_location = carla.Location(x=target_x, y=target_y, z=1)
 
    blueprintsTarget = world.get_blueprint_library().filter("vehicle.diamondback.century")
    blueprintsTarget = world.get_blueprint_library().filter("vehicle.*")
    target_bp = random.choice(blueprintsTarget)

    target_bp = world.get_blueprint_library().filter("walker.pedestrian.0050")[0]

    target_transform = carla.Transform(target_location, carla.Rotation(yaw=vehicle_yaw*180/np.pi + 90))
    world.spawn_actor(target_bp,target_transform)
    print("spawned target with transform: ", target_transform)


    camera_list = actor_list.filter('sensor.camera.rgb')
    spectator = world.get_spectator()
    spectator.set_transform(camera_list[1].get_transform())

   
if __name__ == '__main__':
    main()