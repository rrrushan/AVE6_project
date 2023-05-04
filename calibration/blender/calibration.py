import numpy as np
import cv2 as cv
import glob

points_row = 10
points_col = 7

objp = np.zeros((points_row * points_col, 3), np.float32)
objp[:,:2] = np.mgrid[0:points_row, 0:points_col].T.reshape(-1,2)
objp *= 100

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# # original images
# images = glob.glob('Blender/01_Rendered_images/*.png')

# # cropped images
# images = glob.glob('Blender/02_Cropped/*.png')

# # cropped and padded images
# images = glob.glob('Blender/03_Cropped_Padded/*.png')

# cropped, padded and resized images
images = glob.glob('Blender/04_Resized/*.png')

images.sort()
for i, fname in enumerate(images):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(f"Processing image {i}")
    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (points_row, points_col), None)
   
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (points_row, points_col), corners, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

print('Image point arrays: ', len(imgpoints))
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)

# def write_result(file_name, ret, mtx, dist, rvecs, tvecs, shape):
#     with open(f'Blender/{file_name}.txt', 'a') as f:
#         f.write(f'Image shape:\n')
#         f.write(f'{shape}\n')
#         f.write(f'Error:\n')
#         f.write(f'{ret}\n')
#         f.write(f'Intrinsics:\n')
#         f.write(f'{mtx}\n')
#         f.write(f'Extrinsics:\n')
#         for i in range(len(tvecs)):
#             f.write(f'Image {i}:\n')
#             f.write(f't:\n{tvecs[i]}\n')
#             f.write(f'r:\n{rvecs[i]}\n')
#         f.write(f'Distortion:\n')
#         f.write(f'{dist}')

# filename = '23_04_27'
# write_result(filename, ret, mtx, dist, rvecs, tvecs, img.shape)