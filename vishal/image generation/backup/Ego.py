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
        ego_data = ('tesla', 30.0182, 133.947205, -179.679535)
        blueprint_library = world.get_blueprint_library()

        # Filter all the blueprints of type (e.g. 'audi') 
        bp = random.choice(blueprint_library.filter(ego_data[0]))
        
        # For vehicles, a color can be selected 
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)     

        # Now we need to give an initial transform to the vehicle
        ego_transform = carla.Transform(carla.Location(x=ego_data[1], y=ego_data[2], z=0.6), 
                                        carla.Rotation(pitch=0.0, yaw=ego_data[3], roll=0.0))
        print('spawn point selected at: ', ego_transform)

        ego = world.spawn_actor(bp, ego_transform)
        actor_list.append(ego)
        actor_name_list[ego_data[0]] = ego.id
        print('created %s' % ego.type_id)
        spectator.set_transform(ego_transform)

        time.sleep(5)

    finally:
    
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()
