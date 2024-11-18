import cv2
import matplotlib.pyplot as plt
from pathlib import Path
IMAGE_PATH = Path.cwd() / 'images' / 'tennis.jpg'

#Example from  chatGPT
# Step 1: Read the image
image = cv2.imread(IMAGE_PATH)

# Step 2: Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Canny Edge Detection
edges = cv2.Canny(gray_image, 100, 200)

# Step 4: Display the original and edge-detected images using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detected Image')

plt.show()

"""
The example you’ve seen, using edge detection, is a fundamental technique in computer vision that can 
be applied to several real-world use cases. Here are some applications where edge detection like Canny can be used:

1. Object Detection and Recognition
Application: Detecting the shape of objects in an image, such as cars, people, or everyday items.
Example: In self-driving cars, edge detection helps recognize road signs, lanes, and obstacles by detecting
the boundaries of different objects. The sharp changes in pixel intensity represent the edges that can define lanes 
or separate objects from the background.

2. Image Segmentation
Application: Splitting an image into meaningful regions.
Example: In medical imaging, edge detection is used to segment different structures like tumors or organs from medical images
like X-rays, MRIs, or CT scans. The sharp boundaries between tissues can help doctors and AI systems locate regions of interest.

3. Face Detection
Application: Identifying facial features like eyes, nose, and mouth in images.
Example: Edge detection can assist in extracting key facial features, which can be fed into more advanced algorithms to
detect and recognize faces. Many smartphones use this concept as part of face unlock systems.

4. Barcode and QR Code Scanning
Application: Detecting the boundaries of a barcode or QR code for scanning purposes.
Example: Supermarket scanners use edge detection to quickly recognize and process the edges of barcodes on products.
Similarly, QR code scanning apps use edge detection to identify the code's boundary.

5. Lane Detection in Autonomous Vehicles
Application: Identifying road lanes in real-time.
Example: Self-driving cars rely on edge detection to track the road lane markings. By detecting the edges of the lane, 
the vehicle can stay within its designated lane or make safe lane changes.

6. Object Counting
Application: Counting objects in images based on their shapes and edges.
Example: In manufacturing or quality control, edge detection can help count the number of items (e.g., bottles on 
a conveyor belt) or identify defective products by highlighting irregular edges.

7. Document Scanning
Application: Detecting and cropping document boundaries.
Example: Document scanning apps (like those on smartphones) use edge detection to find the borders of a paper in a photo 
and automatically crop the image, making it ready for digital storage.

8. Optical Character Recognition (OCR)
Application: Preprocessing step for extracting text from images.
Example: Before extracting text from an image, edge detection helps identify the boundaries of text regions, improving 
the accuracy of character recognition software.

9. Robotic Vision
Application: Helping robots “see” and interact with objects.
Example: In manufacturing or assembly lines, robots can use edge detection to identify the shapes of objects they need to
pick up, manipulate, or assemble.

10. Image Stitching
Application: Combining multiple images into a panorama.
Example: When creating panoramic photos, edge detection helps identify key points between overlapping images, making it easier
to stitch them together seamlessly.

Edge detection is often the starting point for these applications. The detected edges are then used for higher-level 
tasks like classification, feature extraction, and decision-making.

"""