import glob
import os
import sys

try:
    # specify the path to the *.egg file
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import math
import numpy as np

def main():
    actor_list = []
    actor_name_list = {}

    try:
        # Connect to the world
        client = carla.Client('localhost', 2000)
        client.set_timeout(5)

        # Once we have a client we can retrieve the world that is currently running
        world = client.get_world()
        spectator = world.get_spectator()
      
        '''SPAWNING THE EGO VEHICLE'''
        option = 1
        ego_type = 'bmw'
        ego_data = (ego_type, 30.0182, 133.947205, -179.679535)
        blueprint_library = world.get_blueprint_library()

        # Filter all the blueprints of type (e.g. 'audi') 
        bp = random.choice(blueprint_library.filter(ego_data[0]))
        
        # For vehicles, a color can be selected 
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)   
              
        if option == 0:
        # Now we need to give an initial transform to the vehicle
            ego_transform = carla.Transform(carla.Location(x=ego_data[1], y=ego_data[2], z=0.6), 
                                            carla.Rotation(pitch=0.0, yaw=ego_data[3], roll=0.0))
            print('spawn point selected at: ', ego_transform)
            print()
        else:
            ego_transform = random.choice(world.get_map().get_spawn_points())
            print('spawn point selected at: ', ego_transform)
            print()

        ego = world.spawn_actor(bp, ego_transform)
        actor_list.append(ego)
        actor_name_list[ego_data[0]] = ego.id
        print('created %s' % ego.type_id, '\n')
        spectator.set_transform(ego_transform)

        '''SPAWNING ACTORS'''
        no_actors = 5

        # defining the spawning area
        spawn_range = [20, 30]
        margin = 5
        cosine = math.cos(math.radians(ego_transform.rotation.yaw))
        sine = math.sin(math.radians(ego_transform.rotation.yaw))


        'CORRECT MATRIX ROTATION'
        rel_coor_1 = [-spawn_range[0]//2 + margin, 0, 0.6]
        rel_coor_2 = [ spawn_range[0]//2 + margin, 0, 0.6]
        rel_coor_3 = [-spawn_range[0]//2 + margin, spawn_range[1], 0.6]
        rel_coor_4 = [ spawn_range[0]//2 + margin, spawn_range[1], 0.6]
        # test_point = [ 1, 1, 1]

        tf_matrix = np.array([ [cosine, -sine,  0],
                               [sine,   cosine, 0],
                               [0,      0,      1]
                            ])

        print(tf_matrix)

        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_z = 0

        ego_coords = np.array([
                                ego_x,
                                ego_y,
                                ego_z,
                                ])

        coordinate_1 = ego_coords + np.dot(rel_coor_1,  tf_matrix)
        coordinate_2 = ego_coords + np.dot(rel_coor_2,  tf_matrix)
        coordinate_3 = ego_coords + np.dot(rel_coor_3,  tf_matrix)
        coordinate_4 = ego_coords + np.dot(rel_coor_4,  tf_matrix)
        print('1: ', np.dot(rel_coor_1,  tf_matrix))
        print('2: ', coordinate_1)

        'INCORRECT MATRIX ROTATION'
        # coordinate_1 = [ego_transform.location.x, ego_transform.location.y]
        # coordinate_2 = [coordinate_1[0] + cosine * spawn_range, coordinate_1[1] + sine*spawn_range] 
        # coordinate_3 = [coordinate_1[0] + math.sqrt(spawn_range**2 + spawn_range**2)*cosine, 
        #                 coordinate_1[1] + math.sqrt(spawn_range**2 + spawn_range**2)*sine]
        # coordinate_4 = [coordinate_1[0] - cosine * spawn_range, coordinate_1[1] + sine*spawn_range]
        print('\n', 'Area of spawning for non-ego actors:', coordinate_1, coordinate_2, coordinate_3, coordinate_4, '\n')
        print()

        min_x = min(coordinate_1[0], coordinate_2[0], coordinate_3[0], coordinate_4[0])
        max_x = max(coordinate_1[0], coordinate_2[0], coordinate_3[0], coordinate_4[0])
        min_y = min(coordinate_1[1], coordinate_2[1], coordinate_3[1], coordinate_4[1])
        max_y = max(coordinate_1[1], coordinate_2[1], coordinate_3[1], coordinate_4[1])
        print('min & max: x & y', min_x, max_x, min_y, max_y, '\n')
        print()
        
        spawn_points = world.get_map().get_spawn_points()

        # filtering out the spawning points
        filtered_spawn_points = [ spawn_point for spawn_point in spawn_points 
                                if min_x <= spawn_point.location.x <= max_x
                                and min_y <= spawn_point.location.y <= max_y
                                ]
        # for spawn_point in filtered_spawn_points:
        #     print(spawn_point, '\n')

        print()
        print('The number of filtered spawn points', len(filtered_spawn_points))
        print()

        random.shuffle(filtered_spawn_points)
        
        # spawning vehicles
        iterations = min(no_actors, len(filtered_spawn_points), 2)
        actor_type = 'model3'

        for i in range(iterations):
            actor_data = (actor_type, filtered_spawn_points[i].location.x, filtered_spawn_points[i].location.y, 
                          filtered_spawn_points[i].rotation.yaw)


            bp = random.choice(blueprint_library.filter(actor_type))
        
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)     

            actor_transform = carla.Transform(carla.Location(x=actor_data[1], y=actor_data[2], z=0.6), 
                                            carla.Rotation(pitch=0.0, yaw=actor_data[3], roll=0.0))
            print('The spawn point for actor ', i, ' is selected at: ', actor_transform.location)
            
            actor = world.try_spawn_actor(bp, actor_transform)
            actor_list.append(actor)
            actor_name_list[i] = actor.id
            print('created %s' % actor.type_id)
            time.sleep(1)

        time.sleep(10)

    finally:
    
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()
