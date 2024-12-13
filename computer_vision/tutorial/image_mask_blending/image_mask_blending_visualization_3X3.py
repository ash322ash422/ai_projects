import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a 3x3 original image with RGB values
# Each pixel will have 3 values (R, G, B)
image_np = np.array([[[100, 150, 200], [50, 60, 70], [30, 40, 50]], 
                     [[200, 220, 240], [10, 20, 30], [70, 80, 90]], 
                     [[120, 130, 140], [180, 190, 200], [255, 255, 255]]], dtype=np.uint8)

# Step 2: Create a binary mask that highlights some of the pixels
# True means the pixel is part of the object to be highlighted
# Here, True indicates that the pixel will be highlighted, while False means the pixel will
# remain unchanged. In this case, we are highlighting pixels at positions [0, 0], [1, 1], 
# and [2, 2].
mask_np = np.array([[True, False, False],
                    [False, True, False],
                    [False, False, True]])

# Step 3: Define a color to overlay on the masked regions (e.g., red)
color = np.array([255, 0, 0], dtype=np.uint8)  # Red color

# Step 4: Blend the mask with the original image
# We apply the mask and blend the original pixel values with the chosen color
modified_image_np = image_np.copy()  # Copy the original image so we can modify it
modified_image_np[mask_np] = modified_image_np[mask_np] * 0.5 + color * 0.5
print("modified_image_np=\n",modified_image_np)
print("modified_image_np[mask_np]=\n",modified_image_np[mask_np])

# Step 5: Visualize the original and modified images side by side with RGB labels

# Set up the figure and axes
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axes[0].imshow(image_np)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Overlay RGB labels on the original image
for i in range(image_np.shape[0]):
    for j in range(image_np.shape[1]):
        rgb_value = image_np[i, j]
        axes[0].text(j, i, f'{rgb_value}', ha='center', va='center', color='white', fontsize=8, 
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

# Modified image
axes[1].imshow(modified_image_np)
axes[1].set_title("Modified Image with Mask")
axes[1].axis('off')

# Overlay RGB labels on the modified image
for i in range(modified_image_np.shape[0]):
    for j in range(modified_image_np.shape[1]):
        rgb_value = modified_image_np[i, j]
        axes[1].text(j, i, f'{rgb_value}', ha='center', va='center', color='white', fontsize=8, 
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

# Show the images
plt.show()
"""
modified_image_np=
 [[[177  75 100]
  [ 50  60  70]
  [ 30  40  50]]

 [[200 220 240]
  [132  10  15]
  [ 70  80  90]]

 [[120 130 140]
  [180 190 200]
  [255 127 127]]]
modified_image_np[mask_np]=
 [[177  75 100]
 [132  10  15]
 [255 127 127]]
"""