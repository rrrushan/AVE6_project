import numpy as np

'''
This script computes the coordinates of the spawn point of the ego vehicle inside the 
Digital Twin. Since the sensors in CARLA are placed 2m in front of the vehicle, to 
spawn the sensors at the same coordinates as the real sensors were placed in the real 
test, the vehicle spawn point must be computed.
'''


x = -35.62
y = -32.45

dist = 2 # spawn point of the sensors with respect to the vehicle
angle = -147.0 + 180 # yaw angle of the vehicle required to place it straight on the road

x_dist = dist * np.cos(angle * np.pi / 180)
y_dist = dist * np.sin(angle * np.pi / 180)

print(x + x_dist)
print(y + y_dist)