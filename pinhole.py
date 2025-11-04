#!/usr/bin/env python3

import os
import math
import numpy as np
import cv2
import piexif # for parsing EXIF metadata

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


# BACKPROJECTION map to ground plane

# Get a sample of all pixel coordinates
num_px = int(h_img)*int(w_img)

max_points = 100000
max_step = num_px / float(max_points)
max_step = math.sqrt(max_step) # account for 2D grid - 2 dimensions
step = max(1, int(math.ceil(max_step))) # use ceiling and int to get integer step

x_coords = np.arange(0, w_img, step, dtype=np.float64) # sampled column indices of original image
y_coords = np.arange(0, h_img, step, dtype=np.float64) # sampled row indices of original image
xs, ys = np.meshgrid(x_coords, y_coords, indexing='xy') # grid of sampled pixel coordinates

# use ravel to ensure 1D arrays
xs = xs.ravel()
ys = ys.ravel()

ones = np.ones_like(xs)
pix_H = np.stack([xs, ys, ones], axis=0) # homogeneous pixel coords for mapping

d_c = Kinv @ pix_H # direction in camera coords
d_w = R.T @ d_c # direction in world coords

# get scale s for interesection with ground plane
d_wz = d_w[2, :] # z component of direction in world coords

# ensure dz is not a zero vector
d_wz_nonzero = np.where(d_wz == 0, 1e-10, d_wz)

s = -t[2] / d_wz_nonzero # scale necessary to intersect ground plane (z=0)

# only keep points where s > 0 (in front of camera)
s = np.where(s > 0, s, 0)

# ground points in world coords X_w(s) = C + s * d_w
X = t[0] + s * d_w[0, :]
Y = t[1] + s * d_w[1, :]
Z = t[2] + s * d_w[2, :] # should be all zeros in ground plane

pts_ground = np.stack([X, Y, Z], axis=1) # Nx3 array of ground points


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

Xc = t[0] + s_c * d_w_corners[0, :]
Yc = t[1] + s_c * d_w_corners[1, :]

xmin, xmax = float(np.min(Xc)), float(np.max(Xc)) # ground corners
ymin, ymax = float(np.min(Yc)), float(np.max(Yc))

# add asymmetric padding: give much more room at the top and extra on the sides to fit whole image (could be optimized)
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
# details from https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga6f6f4f5f1a4f5aa3f6b2c5f1e3c8b8e1

# save orthophoto image
# change back to BGR for saving with OpenCV
orthophoto = cv2.cvtColor(orthophoto, cv2.COLOR_RGB2BGR)
output_filename = r"C:\Users\tpbar\OneDrive\Desktop\Vision_Project\Hazard-Distance-Detection\Image_Version\Camera\output_ortho.png"
cv2.imwrite(output_filename, orthophoto)
print("Successful.")