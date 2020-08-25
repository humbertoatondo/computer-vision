import numpy as np
import cv2
import glob
import yaml
import os

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
obj_points = [] # 3d point in real world space
img_points = [] # 2d points in image plane.

images = glob.glob(r'images/*.jpeg')

# Create new directory for image results in calibration.
path = os.getcwd()
results_dir = path + "/calibration_results"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

found = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        img_points.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,7), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
        
        # Save results from calibration.
        image_name = results_dir + '/result' + str(found) + '.png'
        cv2.imwrite(image_name, img)

        found += 1

print("Number of images used for calibration: ", found)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1],None,None)

img = cv2.imread('test.jpeg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))





# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)




# Re-projection Error
mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i],img_points2, cv2.NORM_L2)/len(img_points2)
    mean_error += error

print("total error: ", mean_error/len(obj_points))