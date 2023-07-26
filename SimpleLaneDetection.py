import cv2
import numpy as np
import os
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

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
print(edges.shape)

# Use morphological transformation (Dilation) to improve edge detection
kernel = np.ones((3,3), np.uint8)  # Define the structure for morphological operation
edges_dilated = cv2.dilate(edges, kernel, iterations=1)  # Dilate to reinforce the lines

# Skeletonize the image
edges_dilated = edges_dilated / 255  # Normalize to 0,1
skeleton = skeletonize(edges_dilated)
skeleton = (skeleton * 255).astype(np.uint8)  # Convert back to 0-255 scale
skeleton1 = skeleton.copy()
print(skeleton1.shape)

# Crop the top part of the image by 250 pixels
tl = (250, 274-250)
tr = (370, 274-250)
bl = (2, 402-250)
br = (623, 393-250)
# Draw circles on the cropped image
cv2.circle(cropped_img, tl, 5, (255,0,255), -1)
cv2.circle(cropped_img, tr, 5, (255,0,255), -1)
cv2.circle(cropped_img, bl, 5, (255,0,255), -1)
cv2.circle(cropped_img, br, 5, (255,0,255), -1)

# Define source points for perspective transformation
src_pts = np.float32([tl,bl,tr,br])

# Destination points for bird's eye view
dst_pts_BEV = np.float32([[50,50], [50,625],[166,50],[166,625]])

M_BEV = cv2.getPerspectiveTransform(src_pts, dst_pts_BEV)

# Apply perspective transformation to the cropped image
warped_BEV = cv2.warpPerspective(skeleton1, M_BEV, (230, 625))


# Extract the shape of the image
image_shape = warped_BEV.shape
image_height, image_width = image_shape[:2]

# Define the width of the graph
graph_width = 400

# Calculate the scaling factor to map the image width to the graph width
scale_factor = graph_width / image_width

# Calculate the position where half of the image corresponds to position 0 on the graph
center_offset = image_width // 2

# Define the x-axis offset adjustment (in pixels)
offset_adjustment = 0  # Adjust this value to shift the image left or right

# Calculate the offset based on the scaling factor
offset = offset_adjustment / scale_factor

# Split the image into two halves
left_image = warped_BEV[:, :center_offset]
right_image = warped_BEV[:, center_offset:]

# Plotting the bird-eye view (BEV) images on a graph
fig, ax = plt.subplots()

# Plot the left half of the image mirrored on the graph with the adjusted offset
ax.imshow(np.flip(left_image, axis=1), cmap='gray', extent=(-graph_width/2 - offset, -offset, image_height, 0))

# Plot the right half of the image on the graph with the adjusted offset
ax.imshow(right_image, cmap='gray', extent=(offset, graph_width/2 + offset, image_height, 0))

# Plot bird at the position (0,0)
bird_pos = (0, 0)
ax.plot(bird_pos[1], bird_pos[0], 'ro')

# Plot lanes on -x, x axes
lane = np.linspace(-image_height//8, image_height//8, num=image_height)
ax.plot(-lane, np.zeros_like(lane), 'g-')
ax.plot(lane, np.zeros_like(lane), 'g-')

plt.show()

# Display the original image, the cropped image, the edges, dilated edges, the skeleton, and the warped image
cv2.imshow("Original Image", img)
cv2.imshow("Cropped Image", cropped_img)
cv2.imshow("Edges", edges)
cv2.imshow("Dilated Edges", edges_dilated*255)
cv2.imshow("Skeleton", skeleton1)
cv2.imshow("BEV", warped_BEV)
cv2.waitKey(0)
cv2.destroyAllWindows()




