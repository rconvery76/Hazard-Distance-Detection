###############################################################
def createDisparityMap(imgA, imgB):
  stereo = cv2.StereoBM_create(numDisparities=1024, blockSize=5)
  disparityMap = stereo.compute(imgA, imgB)

  return disparityMap

def findDepth(dMap, B, f, percentile=99):
  # Un-scale by 16 (auto scaled by StereoBM)
  disp = dMap.astype(np.float32) / 16.0

  # Calculate depth using a top percentile (default 99th) of disparity
  # This account for noise that can give false depth estimates
  top_disp = np.percentile(disp, percentile)
  depth = f * B / top_disp

  return depth

###############################################################
if __name__ == "__main__":
  import sys
  import cv2
  import numpy as np
  import matplotlib.pyplot as plt

  if len(sys.argv) != 3:
    print("Usage: python disparitymap.py <left_img.jpg> <right_img.jpg>")
    sys.exit(1)

  # Baseline distance in meters
  B = 1.98

  # Focal length in pixels
  f = 1030.21

  # Open image files
  fileA = sys.argv[1]
  fileB = sys.argv[2]
  imgA = cv2.imread(fileA, cv2.IMREAD_GRAYSCALE)
  imgB = cv2.imread(fileB, cv2.IMREAD_GRAYSCALE)
  if imgA is None or imgB is None:
    print("Error: Failed to load images.")
    exit(1)
  elif imgA.shape != imgB.shape:
    print("Error: Images are not the same size.")
    exit(1)

  # Create and display disparity map
  disparityMap = createDisparityMap(imgA, imgB)
  plt.imshow(disparityMap, 'gray')
  plt.show()

  # Calculate depth
  depth = findDepth(disparityMap, B, f)
  print(f"Distance to closest object: {depth:.1f} meters")
