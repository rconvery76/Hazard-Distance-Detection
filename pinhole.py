#!/usr/bin/env python3

import os
import math
import numpy as np
import cv2

# image file management
image_file = r"C:\Users\tpbar\OneDrive\Desktop\Vision_Project\Hazard-Distance-Detection\Image_Version\Camera\1758997612945.jpg"

# camera extrinsics
cam_height = 5 * 0.3048 # camera height in meters (5 feet)

# pitch is rotation about the x axis that maps world->camera.
pitch_deg = math.radians(-70.0)  # -70 => 70 up from horizontal = 20 downfrom vertical

t = np.array([0.0, 0.0, float(cam_height)], dtype=float).T # camera center in world coords
R = np.array([[1, 0, 0],
              [0, math.cos(pitch_deg), -math.sin(pitch_deg)],
              [0, math.sin(pitch_deg), math.cos(pitch_deg)]], dtype=float) # rotation about x axis

'''
Use -70 because R is from world to camera coords, so negative pitch tilts camera 'up' relative to ground plane
'''

T = -R @ t # translation vector from world to camera coords

# camera intrinsics

'''
K = [[1030.21,    0.0, 640.0],
     [   0.0, 1030.21, 360.0],
     [   0.0,    0.0,   1.0]]
'''

focal_length = 1030.21 # focal length in pixels
sensor_width = 3.99 # sensor width in mm

cx = 1280 / 2.0
cy = 720 / 2.0
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0,   0,  1]], dtype=float)
Kinv = np.linalg.inv(K)

# read image (defaults to BGR format)
img = cv2.imread(image_file)
if img is None:
    print("Failed to read image:", image_file)
    exit(1)
    
h_img, w_img, _ = img.shape
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB


# Create ORTHOPHOTO image via homography
# H = K [R | T] t

print("Creating homography")
# construct homography from ground plane (Z=0): H = K [r1 r2 t]
H = K @ np.hstack([R[:, 0].reshape(3, 1), R[:, 1].reshape(3, 1), t.reshape(3,1)])  # 3x3, columns are 1 and 2 of R, and t


# create output orthophoto image grid
ortho_scale = 0.005 # 5 millimeters per pixel
ortho_width = int(w_img)
ortho_height = int(h_img)
ortho_img = np.zeros((ortho_height, ortho_width, 3), dtype=np.uint8)

# the orthophoto grid MUST cover an area big enough to include the ground points
# compute ground-plane size by backprojecting the image corners to the ground
corners_pix = np.array([[0.0, 0.0, 1.0],
                        [w_img - 1.0, 0.0, 1.0],
                        [w_img - 1.0, h_img - 1.0, 1.0],
                        [0.0, h_img - 1.0, 1.0]]).T  # 3x4

d_c_corners = Kinv @ corners_pix                    # camera directions
d_w_corners = R.T @ d_c_corners                      # world directions
d_wz_c = d_w_corners[2, :]
d_wz_c_safe = np.where(np.abs(d_wz_c) < 1e-9, 1e-9, d_wz_c) # avoid div by zero
s_c = -t[2] / d_wz_c_safe
s_c = np.where(s_c > 0, s_c, 0.0) # only keep positive scales, in front of camera

Xc = t[0] + s_c * d_w_corners[0, :] # x values of ground points
Yc = t[1] + s_c * d_w_corners[1, :] # y values of ground points

xmin, xmax = float(np.min(Xc)), float(np.max(Xc)) # ground corner values
ymin, ymax = float(np.min(Yc)), float(np.max(Yc))

# add symmetric padding: give much more room at the top and extra on the sides to fit whole image (not optimized based on camera rotation, but suffices as hardcoded)
padding = 2.5
xmin -= padding
xmax += padding
ymin -= padding
ymax += padding

# compute orthophoto pixel dimensions from world extents
ortho_width = max(1, int(np.ceil((xmax - xmin) / ortho_scale))) # at least 1 pixel wide. 
ortho_height = max(1, int(np.ceil((ymax - ymin) / ortho_scale)))

# create world grid (X_world, Y_world) matching orthophoto pixels
xs = np.arange(ortho_width, dtype=np.float64)
ys = np.arange(ortho_height, dtype=np.float64)
Xg, Yg = np.meshgrid(xs, ys, indexing='xy')   # orthophoto grid (cols, rows)

X_world = xmin + (Xg.ravel() * ortho_scale) 
Y_world = ymin + (Yg.ravel() * ortho_scale)
ones_world = np.ones_like(X_world)
# create 3xN homogeneous world points (X, Y, 1) so H can multiply them
world_pts = np.stack([X_world, Y_world, ones_world], axis=0)  # 3xN homogeneous coords

# project using H (3x4)
projection = H @ world_pts

# copilot help for these lines, preparing for image creation via cv2.remap:

# normalize homogeneous coordinates 
w = projection[2, :]
w_nonzero = np.where(w == 0, 1e-10, w) # avoid div by zero
proj = projection / w_nonzero

# map to original image pixel coordinates and reshape to orthophoto grid
xs_proj = proj[0, :].reshape(ortho_height, ortho_width).astype(np.float32) # get x (column) coordinates
ys_proj = proj[1, :].reshape(ortho_height, ortho_width).astype(np.float32)

# create a 2-channel map (HxWx2) for cv2.remap
map_proj = np.stack([xs_proj, ys_proj], axis=-1).astype(np.float32)

# create orthophoto image via remapping with cv2.remap (provide 2-channel map and map2=None)
orthophoto = cv2.remap(img_rgb, map_proj, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
# details from https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html

# save orthophoto image
# change back to BGR for saving with OpenCV
orthophoto = cv2.cvtColor(orthophoto, cv2.COLOR_RGB2BGR)
output_filename = r"C:\Users\tpbar\OneDrive\Desktop\Vision_Project\Hazard-Distance-Detection\Image_Version\Camera\output_ortho.png"
cv2.imwrite(output_filename, orthophoto)
print("Successful.")