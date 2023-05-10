import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

points_row = 10
points_col = 7

objp = np.zeros((points_row * points_col, 3), np.float32)
objp[:,:2] = np.mgrid[0:points_row, 0:points_col].T.reshape(-1,2)
objp *= 500

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# cropped, padded and resized images
images = glob.glob('calibration/Carla/Calibration_Images/23_05_10/*.png')
# count = 0

for i, fname in enumerate(images):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(f"Processing image {i}", fname)
    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (points_row, points_col), None)
   
    if ret == True:
        objpoints.append(objp)
        # corners2 = cv.cornerSubPix(gray,corners, (5,5), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (points_row, points_col), corners, ret)
        cv.imshow('img', img)
        # cv.imwrite('/home/carla/AVE6_project/calibration/Carla/Calibration_Images/calib_results/%.6d.png' % count, img)
        # count += 1 
        cv.waitKey(500)

print('Image point arrays: ', len(imgpoints))
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)

def write_result(file_name, ret, mtx, dist, shape):
    with open(f'{file_name}.txt', 'a') as f:
        f.write(f'Image shape:\n')
        f.write(f'{shape}\n')
        f.write(f'Error:\n')
        f.write(f'{ret}\n')
        f.write(f'Intrinsics:\n')
        f.write(f'{mtx}\n')
        f.write(f'Distortion:\n')
        f.write(f'{dist}')

filename = 'calibration/Carla/07_05_23'
write_result(filename, ret, mtx, dist, img.shape)