import cv2
import numpy as np
import sqlite3
import os
import uuid
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
import torch
from pathlib import Path
IMAGE_PATH = Path.cwd() / 'images' / 'busy_street_scene_with_multiple_objects_like_cars_people_traffic_lights_buildings.jpg'

# Set up the segmentation model (using pre-trained Mask R-CNN)
weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1  # or use .DEFAULT for the latest weights
model = maskrcnn_resnet50_fpn(weights=weights)
model.eval()


# Transform to Tensor
#Transformation Pipeline:

# The T.Compose function allows you to combine multiple transformations in sequence. In this case, you only have one 
# transformation: T.ToTensor().
# T.ToTensor():

# - This transformation takes an image (in PIL format or NumPy array format) and converts it to a PyTorch tensor. A tensor 
# is a multi-dimensional array used in PyTorch for performing computations.
# - Converting the image to a tensor involves:
#   - Rearranging the image dimensions from (Height, Width, Channels) to (Channels, Height, Width).
#   - Converting the pixel values from integers (ranging from 0 to 255) to floating-point numbers (ranging from 0.0 to 1.0).

# Purpose of the Transformation:
# PyTorch models expect images to be in tensor format, so the ToTensor() function prepares the image for model inference by
# converting it into a suitable format.
transform = T.Compose([T.ToTensor()])

# Create directories for saving images and metadata
os.makedirs('extracted_objects', exist_ok=True)

# Step 1: Database setup
def create_db():
    conn = sqlite3.connect('image_metadata.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS Objects (
                      object_id TEXT PRIMARY KEY, 
                      master_id TEXT, 
                      filepath TEXT)''')
    conn.commit()
    conn.close()

# Step 2: Save metadata to the database
def save_metadata(object_id, master_id, filepath):
    conn = sqlite3.connect('image_metadata.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO Objects (object_id, master_id, filepath) 
                      VALUES (?, ?, ?)''', (object_id, master_id, filepath))
    conn.commit()
    conn.close()

# Step 3: Segment the image and extract objects
def extract_objects(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # A unique identifier (UUID) is generated for the original image. This ID will help track all the objects that are 
    # extracted from this particular image. UUIDs are universally unique, ensuring that each image has a distinct ID, 
    # useful when handling large datasets.
    master_id = str(uuid.uuid4())  # Unique master ID for the original image
    image_tensor = transform(image).unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        prediction = model(image_tensor)[0] # the [0] selects the predictions for the first image in the batch

    # prediction['masks']: This contains the segmentation masks for all detected objects in the image. Each mask
    # is a probability map showing the likelihood of each pixel belonging to an object.
    for i, mask in enumerate(prediction['masks']):
        # mask[0] selects the first channel of the mask, which is the binary mask itself (not the batch dimension).
        # .mul(255) multiplies the mask values by 255. Originally, the values in the mask are probabilities between 0 and
        # 1 (e.g., 0.95 means the pixel has a 95% chance of being part of the object). Multiplying by 255 converts it to
        # an 8-bit format suitable for further processing (like using it as an image mask).
        # .byte() converts the mask to an 8-bit integer type.
        # .cpu() moves the tensor from GPU memory to CPU memory if the model is using GPU. This step is important for subsequent 
        #  operations that rely on NumPy, which works on CPU data.
        # .numpy() converts the PyTorch tensor to a NumPy array for further image processing operations.
        mask_np = mask[0].mul(255).byte().cpu().numpy()
        
        # Binary mask. Converts the mask into a binary mask where pixel values greater than 127 are considered part of the object (True),
        # and the rest are background (False).
        object_mask = mask_np > 127  

        # This sums the True values in the binary mask (which represent object pixels).
        # If the object has fewer than 100 pixels, it is ignored because the mask is too small to be a meaningful object. 
        if np.sum(object_mask) > 100:  # Ignore small masks
            object_id = str(uuid.uuid4())
            contours, _ = cv2.findContours(object_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                cropped_image = np.array(image)[y:y+h, x:x+w]

                object_image_path = f'extracted_objects/object_{object_id}.png'
                Image.fromarray(cropped_image).save(object_image_path)
                save_metadata(object_id, master_id, object_image_path)

    print(f"Extracted objects saved with master ID: {master_id}")

# Step 4: Run extraction and save segmented objects
if __name__ == "__main__":
    create_db()
    image_path = IMAGE_PATH  # Provide your image path here
    extract_objects(image_path)
