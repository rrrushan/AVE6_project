import cv2
import numpy as np

IMG_PATH = "/home/carla/admt_student/team3_ss23/ROS_1/bag_imgs/manual/707.png"

# KiTTi Image shape: 1242 x 375, ar = 3.312
# Our image shape: 1936 x 1464, required_width = 584.54 (585)
TARGET_AR_RATIO = 1242 / 375 # 3.312
TARGET_FOCUS_SCALING = 1676.625 / 1936 # 0.866
TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP = 460
TARGET_RESIZE_WIDTH = 1920

img = cv2.imread(IMG_PATH)
img_orig = img.copy()
print(img.shape)
# total_width = 585
# img = img[int((img.shape[0] - total_width)/2):int((img.shape[0] + total_width)/2), :]
img = img[TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP:img.shape[0], :]
print(img.shape)

h, w, c = img.shape
required_pad_left_right = int((TARGET_AR_RATIO * h - w)/ 2)
img = cv2.copyMakeBorder(img, 0, 0, required_pad_left_right, required_pad_left_right, cv2.BORDER_CONSTANT, None, value = 0)
print(img.shape)

h, w, c = img.shape
img = cv2.resize(img, (TARGET_RESIZE_WIDTH, int(TARGET_RESIZE_WIDTH * h/w)))
box = np.array([(700, 350), (860, 500)])
img = cv2.rectangle(img, box[0], box[1], (255, 0, 0), 2)
print(img.shape)
cv2.imwrite("outputs/707resize_test.png", img)

remove_resize_x, remove_resize_y = box[:, 0] * w / TARGET_RESIZE_WIDTH, box[:, 1] * w / TARGET_RESIZE_WIDTH
remove_resize_x = remove_resize_x - required_pad_left_right
remove_resize_y = remove_resize_y + TARGET_HEIGHT_IGNORANCE_PIXELS_FROM_TOP

final_box_points = np.column_stack((remove_resize_x, remove_resize_y)).astype(int)
print(final_box_points)
img_orig = cv2.rectangle(img_orig, final_box_points[0], final_box_points[1], (255, 0, 0), 2)
cv2.imwrite("outputs/707orig.png", img_orig)