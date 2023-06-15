import numpy as np

x = -35.62
y = -32.45

dist = 3.7
angle = 31

x_dist = dist * np.cos(angle * np.pi / 180)
y_dist = dist * np.sin(angle * np.pi / 180)

print(x + x_dist)
print(y + y_dist)