import cv2
import numpy as np

img = cv2.imread("/home/carla/admt_student/team3_ss23/ROS_1/bag_imgs/manual/44.png")

# cv2.imshow("test", img)
# cv2.waitKey(0)

orig_cam_intrinsic_mtx = np.array([
            [1676.625,   0.0000, 968.0],
            [  0.0000, 1676.625, 732.0],
            [  0.0000,   0.0000,   1.0000]
        ])

# dist_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
dist_coeff = np.array([0.4, 0.135, 0.0, 0.0, 0.0])

output_size = (img.shape[1], img.shape[0])
h, w = img.shape[:2]
mapx, mapy = cv2.initUndistortRectifyMap(orig_cam_intrinsic_mtx, dist_coeff, None, orig_cam_intrinsic_mtx, output_size, cv2.CV_16SC2)
print(mapx.shape, mapy.shape)

dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
mapx = np.linalg.inv(mapx)

cv2.imwrite("distorted_img.png", dst)
cv2.imwrite("orig_img.png", img)