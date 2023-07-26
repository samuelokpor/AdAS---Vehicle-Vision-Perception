import cv2
import numpy as np
import os
from skimage.morphology import skeletonize

IMG_PATH = os.path.join("data", "test5.png")
img = cv2.imread(IMG_PATH)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the cropping boundaries
crop_top_left = (0, 250)  # Top-left point (x, y)
crop_bottom_right = (img_gray.shape[1], img_gray.shape[0])  # Bottom-right point (x, y)

# Crop out the region of interest (ROI)
cropped_img = img_gray[crop_top_left[1]:crop_bottom_right[1], crop_top_left[0]:crop_bottom_right[0]]

# Apply Canny edge detection
edges = cv2.Canny(cropped_img, threshold1=190, threshold2=390)

# Use morphological transformation (Dilation) to improve edge detection
kernel = np.ones((3,3), np.uint8)  # Define the structure for morphological operation
edges_dilated = cv2.dilate(edges, kernel, iterations=1)  # Dilate to reinforce the lines

# Skeletonize the image
edges_dilated = edges_dilated / 255  # Normalize to 0,1
skeleton = skeletonize(edges_dilated)
skeleton = skeleton.astype(np.uint8)  # Convert back to 0,255 scale

# Draw points on the cropped image
skeleton_with_points = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
tl = (222, 222)
bl = (444, 222)
tr = (222, 333)
br = (444, 333)
cv2.circle(skeleton_with_points, tl, 5, (255, 255, 255), -1)
cv2.circle(skeleton_with_points, bl, 5, (255, 255, 255), -1)
cv2.circle(skeleton_with_points, tr, 5, (255, 255, 255), -1)
cv2.circle(skeleton_with_points, br, 5, (255, 255, 255), -1)

# Display the original image, the cropped image, the edges, dilated edges, the skeleton, and the skeleton with points
cv2.imshow("Original Image", img)
cv2.imshow("Cropped Image", cropped_img)
cv2.imshow("Edges", edges)
cv2.imshow("Dilated Edges", edges_dilated * 255)
cv2.imshow("Skeleton", skeleton * 255)
cv2.imshow("Skeleton with Points", skeleton_with_points)
cv2.waitKey(0)
cv2.destroyAllWindows()
