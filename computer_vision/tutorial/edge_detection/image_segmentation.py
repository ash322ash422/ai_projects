import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
IMAGE_PATH = Path.cwd() / 'images' / 'grayscale_MRI_image_human brain_with_circular_tumor.jpg'

# Step 1: Load the image
image = cv2.imread(IMAGE_PATH)

# Step 2: Convert to grayscale.  We load the image and convert it to grayscale because segmentation 
# often works better in single-channel images
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply GaussianBlur to smooth the image (reduces noise)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 4: Perform thresholding to segment the image. We apply binary inverse thresholding,
# which makes the object appear white (255) and the background black (0)
ret, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# Step 5: Find contours of the segmented image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Draw contours on the original image
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Draw in green

# Step 7: Plot original, segmented (thresholded), and contour images
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Thresholded image (segmentation result)
plt.subplot(1, 3, 2)
plt.imshow(thresh, cmap='gray')
plt.title('Thresholded Image (Segmentation)')

# Image with contours. NOTE: To me it looked the same as original image.
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title('Image with Contours')

plt.show()
