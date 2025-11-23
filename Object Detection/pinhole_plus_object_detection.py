import subprocess
import os
import numpy as np
import cv2
import math
import pandas as pd  # For reading the detection results CSV

# Camera Parameters (Ryan)
focal_length = 18144  # in pixels
sensor_width = 4.0  # in mm
cx = 3024 / 2.0  # camera center in x
cy = 4024 / 2.0  # camera center in y
cam_height = 5.167 * 0.3048  # camera height in meters (4ft 2 in)
pitch_deg = math.radians(-10.0)  # pitch angle in radians

# Camera Intrinsics Matrix (K)
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]], dtype=float)

Kinv = np.linalg.inv(K)  # Inverse of K for backprojection

# Camera Extrinsics
t = np.array([0.0, 0.0, cam_height], dtype=float).T  # camera center in world coordinates
R = np.array([[1, 0, 0],
              [0, math.cos(pitch_deg), -math.sin(pitch_deg)],
              [0, math.sin(pitch_deg), math.cos(pitch_deg)]], dtype=float)  # Rotation about X axis

T = -R @ t  # translation vector from world to camera coordinates

# Read labels from labels.txt
def read_labels(labels_file):
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

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

# Backproject Image Pixels to 3D (Ground Plane Intersection)
def backproject_to_ground(img, Kinv, R, t):
    h_img, w_img, _ = img.shape

    # Get a sample of all pixel coordinates
    num_px = int(h_img) * int(w_img)

    max_points = 100000
    max_step = num_px / float(max_points)
    max_step = math.sqrt(max_step)  # Account for 2D grid - 2 dimensions
    step = max(1, int(math.ceil(max_step)))  # Use ceiling and int to get integer step

    x_coords = np.arange(0, w_img, step, dtype=np.float64)  # Sampled column indices of original image
    y_coords = np.arange(0, h_img, step, dtype=np.float64)  # Sampled row indices of original image
    xs, ys = np.meshgrid(x_coords, y_coords, indexing='xy')  # Grid of sampled pixel coordinates

    # Use ravel to ensure 1D arrays
    xs = xs.ravel()
    ys = ys.ravel()

    ones = np.ones_like(xs)
    pix_H = np.stack([xs, ys, ones], axis=0)  # Homogeneous pixel coordinates for mapping

    # Direction in camera coordinates
    d_c = Kinv @ pix_H
    # Direction in world coordinates
    d_w = R.T @ d_c

    # Get scale `s` for intersection with ground plane
    d_wz = d_w[2, :]  # Z component of direction in world coordinates

    # Ensure dz is not zero
    d_wz_nonzero = np.where(d_wz == 0, 1e-10, d_wz)

    s = -t[2] / d_wz_nonzero  # Scale necessary to intersect ground plane (z=0)

    # Only keep points where `s > 0` (in front of the camera)
    s = np.where(s > 0, s, 0)

    # Ground points in world coordinates X_w(s) = C + s * d_w
    X = t[0] + s * d_w[0, :]
    Y = t[1] + s * d_w[1, :]
    Z = t[2] + s * d_w[2, :]  # Should be near zero for ground plane

    return X, Y, Z, xs, ys

# Function to read detections from CSV
def read_detections_from_csv(csv_file):
    detections = pd.read_csv(csv_file)
    boxes = detections[['x1', 'y1', 'x2', 'y2']].values  # Extract bounding box coordinates
    labels = detections['label'].values  # Extract label for each object
    scores = detections['score'].values  # Extract detection score
    return boxes, labels, scores

# Main function to run the pipeline
def main(image_file, labels_file, output_image, output_csv):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_image), exist_ok=True)

    # Run Object Detection
    run_detection(image_file, labels_file, output_image, output_csv)

    # Read the original image
    img = cv2.imread(image_file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Read the detections from the output CSV
    boxes, labels_detected, scores = read_detections_from_csv(output_csv)

    # Backproject to ground and create orthophoto
    X, Y, Z, xs, ys = backproject_to_ground(img, Kinv, R, t)

    h_img, w_img, _ = img.shape
    ortho_scale = 0.005  # 5 millimeters per pixel (can adjust this)
    ortho_width = int(w_img)
    ortho_height = int(h_img)
    ortho_img = np.zeros((ortho_height, ortho_width, 3), dtype=np.uint8)

    # Backproject image and create orthophoto
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

    padding = 2.5
    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding

    ortho_width = max(1, int(np.ceil((xmax - xmin) / ortho_scale)))
    ortho_height = max(1, int(np.ceil((ymax - ymin) / ortho_scale)))

    xs = np.arange(ortho_width, dtype=np.float64)
    ys = np.arange(ortho_height, dtype=np.float64)
    Xg, Yg = np.meshgrid(xs, ys, indexing='xy')   # orthophoto grid (cols, rows)

    X_world = xmin + (Xg.ravel() * ortho_scale) 
    Y_world = ymin + (Yg.ravel() * ortho_scale)
    ones_world = np.ones_like(X_world)
    world_pts = np.vstack([Xg.ravel(), Yg.ravel(), np.ones(Xg.size)])

    H = K @ np.hstack([R, t.reshape(3, 1)])
    projection = H @ world_pts

    w = projection[2, :]
    w_nonzero = np.where(w == 0, 1e-10, w)
    proj = projection / w_nonzero

    xs_proj = proj[0, :].reshape(ortho_height, ortho_width).astype(np.float32)
    ys_proj = proj[1, :].reshape(ortho_height, ortho_width).astype(np.float32)

    map_proj = np.stack([xs_proj, ys_proj], axis=-1).astype(np.float32)

    orthophoto = cv2.remap(img_rgb, map_proj, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    orthophoto = cv2.cvtColor(orthophoto, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image, orthophoto)

    # Calculate distance for each detected object
    for i, (box, label, score) in enumerate(zip(boxes, labels_detected, scores)):
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        x_center = np.clip(int(x_center), 0, w_img - 1)  
        y_center = np.clip(int(y_center), 0, h_img - 1)  

        closest_x_index = np.argmin(np.abs(xs - x_center))
        closest_y_index = np.argmin(np.abs(ys - y_center))

        if closest_x_index < len(X) and closest_y_index < len(Y):
            index = closest_y_index * len(xs) + closest_x_index

            dist = np.sqrt((X[index] ** 2) + (Y[index] ** 2) + (Z[index] ** 2))
            print(f"Object {label} detected at {dist:.2f} meters from the camera.")
        else:
            print(f"Warning: Index out of bounds for object {label} at pixel ({x_center}, {y_center})")

# Example usage
if __name__ == "__main__":
    image_file = "corner_stopsign.jpg"  # Update with your image path
    labels_file = "labels.txt"  # Path to labels.txt
    output_image = "Portrait Photos/Output/output_image.png"  # Path for output image
    output_csv = "output_detections.csv"  # Path for output CSV

    main(image_file, labels_file, output_image, output_csv)
