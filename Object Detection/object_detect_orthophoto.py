#!/usr/bin/env python3

import os
import math
import numpy as np
import cv2
import piexif # for parsing EXIF metadata
import subprocess
import sys
import pandas as pd  # For reading the detection results CSV

# Function to call the external object detection script
def run_detection(image_path, labels_file, output_path, output_csv):
    command = [
        "python", "open_vocab_detect_v2.py",
        "--image", image_path,
        "--labels", labels_file,
        "--out", output_path,
        "--csv", output_csv,
        "--score", "0.3",  # Optional threshold
        "--nms", "0.5",  # Optional NMS threshold
    ]
    
    try:
        print(f"Running detection on {image_path}...")
        subprocess.run(command, check=True)  # Run the detection script
        print(f"Detection complete. Output saved to {output_path} and detections saved to {output_csv}")
    except subprocess.CalledProcessError as e:
        print(f"Error running detection on {image_path}: {e}", file=sys.stderr)

# Read labels from labels.txt
def read_labels(labels_file):
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Function to read detections from CSV
def read_detections_from_csv(csv_file):
    detections = pd.read_csv(csv_file)
    boxes = detections[['x1', 'y1', 'x2', 'y2']].values  # Extract bounding box coordinates
    labels = detections['label'].values  # Extract label for each object
    scores = detections['score'].values  # Extract detection score
    return boxes, labels, scores

# def set_camera_parameters():
#     # camera extrinsics
#     cam_height = 5 * 0.3048 # camera height in meters (5 feet)

#     # pitch is rotation about the x axis that maps world->camera.
#     pitch_deg = math.radians(-70.0)  # -70 => 70 up from horizontal = 20 downfrom vertical

#     t = np.array([0.0, 0.0, float(cam_height)], dtype=float).T # camera center in world coords
#     R = np.array([[1, 0, 0],
#                 [0, math.cos(pitch_deg), -math.sin(pitch_deg)],
#                 [0, math.sin(pitch_deg), math.cos(pitch_deg)]], dtype=float) # rotation about x axis

#     '''
#     Use -70 because R is from world to camera coords, so negative pitch tilts camera 'up' relative to ground plane
#     '''

#     T = -R @ t # translation vector from world to camera coords

#     # camera intrinsics

#     '''
#     K = [[1030.21,    0.0, 640.0],
#         [   0.0, 1030.21, 360.0],
#         [   0.0,    0.0,   1.0]]
#     '''

#     focal_length = 1030.21 # focal length in pixels
#     sensor_width = 3.99 # sensor width in mm

#     cx = 1280 / 2.0
#     cy = 720 / 2.0
#     K = np.array([[focal_length, 0, cx],
#                 [0, focal_length, cy],
#                 [0,   0,  1]], dtype=float)
#     Kinv = np.linalg.inv(K)
#     return K, Kinv, R, t

def set_camera_parameters():
    # camera extrinsics
    cam_height = 5 * 0.3048 # camera height in meters (5 feet)

    # pitch is rotation about the x axis that maps world->camera.
    pitch_deg = math.radians(-66.0)  # -70 => 70 up from horizontal = 20 downfrom vertical

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

    focal_length = 3112.5 # focal length in pixels
    sensor_width = 3.99 # sensor width in mm

    cx = 1125 / 2.0
    cy = 1500 / 2.0
    K = np.array([[focal_length, 0, cx],
                [0, focal_length, cy],
                [0,   0,  1]], dtype=float)
    Kinv = np.linalg.inv(K)
    return K, Kinv, R, t

def back_project(img, Kinv, R, t):
    # BACKPROJECTION map to ground plane

    h_img, w_img, _ = img.shape


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
    return X, Y, Z, xs, ys

def pixel_to_world_ground(x_pix, y_pix, K, Kinv, R, t, debug=False):
    """Convert a pixel coordinate to world ground plane (Z=0) coordinates."""
    pix_H = np.array([x_pix, y_pix, 1.0])
    d_c = Kinv @ pix_H
    d_w = R.T @ d_c
    
    
    # Intersection with ground plane
    if abs(d_w[2]) < 1e-10:
        print(f"      ❌ FAIL: d_w[2] too close to zero (ray parallel to ground)")
        return None, None
    
    s = t[2] / d_w[2]
    
    if s <= 0:
        print(f"      ❌ FAIL: s={s:.3f} <= 0 (ray pointing behind camera)")
        print(f"      This shouldn't happen since d_w[2] > 0 and t[2] > 0")
        print(f"      CHECK: Is t[2] actually the camera height?")
        return None, None
    
    X_world = t[0] + s * d_w[0]
    Y_world = t[1] + s * d_w[1]
    
    return X_world, Y_world

def world_to_ortho_pixel(X_world, Y_world, xmin, ymin, ortho_scale, ymax=None):
    """Convert world ground coordinates to orthophoto pixel coordinates."""
    ortho_x = (X_world - xmin) / ortho_scale
    
    # If ymax is provided, flip Y axis so larger Y_world appears at top of image
    if ymax is not None:
        ortho_y = (ymax - Y_world) / ortho_scale
    else:
        ortho_y = (Y_world - ymin) / ortho_scale
    
    return ortho_x, ortho_y

def create_orthophoto_with_detections(img, K, Kinv, R, t, boxes, labels, scores, output_image, X, Y, Z, xs, ys):
    # Create ORTHOPHOTO image via homography
    h_img, w_img, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
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


    #determine the pixel of the image bottom edge
    print(f"\n[DEBUG] Finding bottom edge of image content...")
    
    gray = cv2.cvtColor(orthophoto, cv2.COLOR_BGR2GRAY)
    
    image_bottom_y = None
    for y in range(ortho_height - 1, -1, -1):
        row = gray[y, :]
        if np.any(row > 1):  # Found non-black pixels
            image_bottom_y = y
            print(f"  Found image content at y={y}")
            break

    # Draw line indicating bottom edge of image content
    cv2.line(orthophoto, (0, image_bottom_y), (ortho_width, image_bottom_y), (255, 0, 255), 2)
    print(f"[DEBUG] Image bottom edge at y={image_bottom_y} pixels in orthophoto.")

    #Process detected boxes to map to orthophoto coordinates
    detection_results = {}
    # Overlay detections onto orthophoto
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box

        bottom_center_x = (x1 + x2) / 2.0
        bottom_center_y = y2

        #print(bottom_center_x, bottom_center_y)

        Xw, Yw = pixel_to_world_ground(bottom_center_x, bottom_center_y, K, Kinv, R, t)
        print(f"Distance of obeject {label} (score: {score:.2f}) forward: {abs(Yw):.2f} meters, {abs(Yw)*39.3701:.2f} inches")

        # if Xw is None or Yw is None:
        #     print(f"Skipping detection {i} due to invalid world coordinates.")
        #     continue
        # ortho_x, ortho_y = world_to_ortho_pixel(Xw, Yw, xmin, ymin, ortho_scale, ymax)
        # ortho_y = ortho_y - (ortho_height - image_bottom_y)  # Flip Y for image coordinates

        # # Draw detection on orthophoto
        # print("Debug for detection")
        # print(f"ortho_y: {ortho_y}, image_bottom_y: {image_bottom_y}, ortho_height: {ortho_height}")
        # cv2.circle(orthophoto, (int(ortho_x), int(ortho_y)), 5, (255, 0, 0), -1)
        # cv2.putText(orthophoto, f"{label}", (int(ortho_x) + 10, int(ortho_y)), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Store new box coordinates
        detection_results[f"{label}_{i}"] = {
            'world_coordinates': (Xw, Yw),
            #'ortho_pixel_coordinates': (ortho_x, ortho_y)
        }

        
    # save orthophoto image
    # change back to BGR for saving with OpenCV
    orthophoto = cv2.cvtColor(orthophoto, cv2.COLOR_RGB2BGR)
    output_filename = "OUTPUT/output_ortho.png"
    cv2.imwrite(output_filename, orthophoto)
    print("Successful.")

    return detection_results, xmin, ymin, ymax, image_bottom_y, ortho_scale

def calculate_distance_to_object(detection_results, img_bottom, object_label):
    for key, data in detection_results.items():
            if object_label in key:
                obj_x, obj_y = data['world_coordinates']
                distance = abs(obj_y - img_bottom) 
                distance *= 0.005  # Convert pixels to meters using ortho_scale
                print(f"Distance to {object_label}: {distance:.2f} meters")
                return distance
            else:
                print(f"No detection found for label: {object_label}")
    

if __name__ == "__main__":
    #Set camera parameters
    K, Kinv, R, t = set_camera_parameters()

    image_file = "INPUT/chair.jpg"
    output_image = "OUTPUT/orthophoto_with_detections.png"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_image), exist_ok=True)


    # read image (defaults to BGR format)
    img = cv2.imread(image_file)
    if img is None:
        print("Failed to read image:", image_file)
        exit(1)

    #Run object detection on an image
    labels_file = "labels.txt"
    output_detection_image = "OUTPUT/detection_output.png"
    output_detection_csv = "OUTPUT/detections.csv"
    run_detection(image_file, labels_file, output_detection_image, output_detection_csv)
    boxes, labels, scores = read_detections_from_csv("OUTPUT/detections.csv")

    #Back project and create orthophoto from image with detections
    X, Y, Z, xs, ys = back_project(img, Kinv, R, t)
    detected_image = cv2.imread(output_detection_image)
    detection_results, xmin, ymin, ymax, image_bottom_y, ortho_scale = create_orthophoto_with_detections(detected_image, K, Kinv, R, t, boxes, labels, scores, output_image, X, Y, Z, xs, ys)

    # print("Detection Results with World and Ortho Coordinates:")
    # for key, data in detection_results.items():
    #     print(f"{key}: World Coords: {data['world_coordinates']}, Ortho Pixel Coords: {data['ortho_pixel_coordinates']}")

    # #Calculate distance to all objects detected
    # for key in detection_results.keys():
    #     label = key.split('_',1)[0]  # Extract label from key
    #     calculate_distance_to_object(detection_results, image_bottom_y, label)

    # #run the ortho photo through edge detection
    # # Load the orthophoto image
    # ortho_img = cv2.imread("OUTPUT/output_ortho.png")
    # if ortho_img is None:
    #     print("Failed to read orthophoto image:", output_image)
    #     exit(1)

    # # Convert to grayscale
    # gray_ortho = cv2.cvtColor(ortho_img, cv2.COLOR_BGR2GRAY)
    # # Apply Canny edge detection
    # edges = cv2.Canny(gray_ortho, 100, 200)
    # # Save the edge-detected image
    # edge_output_image = "OUTPUT/orthophoto_edges.png"
    # os.makedirs(os.path.dirname(edge_output_image), exist_ok=True)
    # cv2.imwrite(edge_output_image, edges)
    # print("Edge-detected orthophoto saved to", edge_output_image)
        



