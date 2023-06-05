import numpy as np
import cv2

# Define the function to draw a single box
def draw_box(img, corners):
    # Convert the corners to integer values
    corners = np.int32(corners)

    # Draw the lines between the corners
    for i in range(0, 4):
        img = cv2.line(img, tuple(corners[i]), tuple(corners[(i+1)%4]), (0, 0, 255), 2)
        img = cv2.line(img, tuple(corners[i+4]), tuple(corners[((i+1)%4)+4]), (0, 0, 255), 2)
        img = cv2.line(img, tuple(corners[i]), tuple(corners[i+4]), (0, 0, 255), 2)

    # Return the image with the box drawn on it
    return img

# Define the list of box coordinates
box_coords = [
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],
    [[2, 2, 2], [2, 3, 2], [3, 3, 2], [3, 2, 2], [2, 2, 3], [2, 3, 3], [3, 3, 3], [3, 2, 3]]
]

# Define the size of the output image
img_size = (800, 600)

# Create a blank image with the specified size and color
img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
img.fill(255)

# Loop through all the boxes and draw them on the image
for corners in box_coords:
    img = draw_box(img, corners)

# Display the resulting image using OpenCV
cv2.imshow("Boxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
