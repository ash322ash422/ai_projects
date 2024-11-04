import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights  # Import weights enum
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
IMAGE_PATH = Path.cwd() / 'images' / 'busy_street_scene_with_multiple_objects_like_cars_people_traffic_lights_buildings.jpg'

# Step 1: Load a pre-trained Mask R-CNN model. It is pre-trained on the COCO dataset
# The COCO (Common Objects in Context) dataset is a large-scale image dataset that is used for training and evaluating computer
# vision models. It contains over 330,000 images with detailed annotations for 80 object categories and 5 captions per image
# Step 1: Load a pre-trained Mask R-CNN model with explicit weights argument
weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1  # or use .DEFAULT for the latest weights
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
model.eval()  # Set the model to evaluation mode. is used when the model is being used for inference 
# (i.e., making predictions without updating its parameters).
# In evaluation mode, several internal settings in the model change:
# - Batch Normalization: Layers like batch normalization (nn.BatchNorm) behave differently in training and evaluation.
# In evaluation mode, batch normalization layers use running averages for mean and variance instead of computing them
# from the current batch of data. This ensures that the model's behavior during inference is consistent.

# - Dropout: Layers like dropout (nn.Dropout) are used during training to randomly disable some neurons to prevent 
# overfitting. In evaluation mode, dropout is turned off, meaning all neurons are active and used for inference. This 
# allows the model to make more stable predictions.

# Therefore, model.eval() ensures the model behaves correctly during inference (predicting bounding boxes and segmentation
# masks) by disabling training-specific behavior like dropout and updating batch normalization stats


# Step 2: Define a function to apply the model to an input image
def segment_image(image_path):
    # Convert("RGB") method ensures that the image is in RGB format, which means that it has 3 color channels
    # (Red, Green, Blue). If the image is in a different format (e.g., grayscale), this step ensures that it 
    # is compatible with models that expect 3-channel inputs
    image = Image.open(image_path).convert("RGB")
    
    # The line weights.transforms() loads the recommended pre-processing transformation for the model based on
    # the pre-trained weights. Pre-trained models often expect images to be pre-processed in specific ways, like 
    # normalization, resizing, or tensor conversion.

    # This transformation typically involves:
    # - Resizing: Rescaling the image to a standard size.
    # - Normalization: Adjusting pixel values to a range or normalizing based on the model's training data (e.g., 
    #   the COCO dataset).
    # - Tensor Conversion: Converting the image from a PIL image to a PyTorch tensor (multi-dimensional array 
    #   used by PyTorch models).
    transform = weights.transforms()  # Use the pre-trained model's recommended transform
    
    # Following applies the transformation to the image and adds a batch dimension to the tensor. 
    # Neural networks in PyTorch expect inputs to be in batches (even if there is only one image). The unsqueeze(0) 
    # adds this batch dimension, changing the shape of the tensor from (C, H, W) (channels, height, width) to 
    # (1, C, H, W).
    # For example: If the image tensor originally has dimensions (Channels, Height, Width) — let’s say (3, 500, 500) — 
    # after applying unsqueeze(0), it becomes (1, 3, 500, 500).
    image_tensor = transform(image).unsqueeze(0)  
    
    # The with torch.no_grad() block tells PyTorch not to compute gradients during this operation. Gradients are only 
    # needed during training when the model is learning, but for inference (i.e., making predictions), gradients are 
    # unnecessary and can slow down computation. Disabling gradients here improves efficiency.
    with torch.no_grad():
        # model(image_tensor) passes the input tensor (which is the transformed image) through the pre-trained Mask R-CNN 
        # model. The model performs object detection and segmentation on the image
        # The model returns a list of predictions for each object detected in the image.
        # Each prediction contains the following information:
        # - Bounding boxes: Coordinates that define the rectangle around each detected object.
        # - Labels: Predicted class labels for each object (e.g., "person", "car", etc.).
        # - Masks: Binary masks that highlight the pixels belonging to each object, allowing for segmentation.
        predictions = model(image_tensor)
    
    return image, predictions[0]  # Return the image and the first batch prediction

# Step 3: Visualize the results (segmented objects)
# A confidence threshold for the object masks. Only masks with a probability greater than 0.5 are 
# considered as "detected"
def visualize_segmentation(image, prediction, threshold=0.5):
    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Get masks, boxes, and labels with a confidence score higher than the threshold
    # Masks:
    # - The prediction['masks'] are probabilities for each pixel in the image, indicating the likelihood that a pixel 
    # belongs to an object.
    # - By using prediction['masks'] > threshold, we binarize these masks, meaning that any pixel with a probability 
    # higher than threshold (0.5 by default) is considered part of the object (the mask becomes True), while the rest are False.
    masks = prediction['masks'] > threshold  # Masks are predicted probabilities; convert to binary mask
    
    # Bounding Boxes:
    # - The prediction['boxes'] contains the coordinates of the bounding boxes for each detected object. These bounding 
    #  boxes are in the format [x_min, y_min, x_max, y_max], where (x_min, y_min) is the top-left corner and 
    # (x_max, y_max) is the bottom-right corner.
    # - The cpu().numpy() operation converts the tensor from GPU format (if it’s on a CUDA device) to a NumPy array 
    # that can be manipulated more easily.
    boxes = prediction['boxes'].cpu().numpy()
    
    # Labels:
    # - prediction['labels'] contains the class labels for each detected object. Similar to bounding boxes, the labels are 
    # converted from a tensor to a NumPy array for further processing.
    labels = prediction['labels'].cpu().numpy()

    # Loop through detected objects
    for i, mask in enumerate(masks):
        # Extract mask and bounding box for each object
        # Extracting the Mask:
        # - Each mask is initially in a 3D tensor shape like [1, H, W]. The squeeze() method removes the extra dimension to 
        #   get a 2D array of shape [H, W], representing a binary mask where True (or 1) indicates pixels belonging to the object.
        # - The mask is then converted to a NumPy array for easy manipulation.
        mask_np = mask.squeeze().cpu().numpy()
        box = boxes[i] # The bounding box corresponding to the current object is extracted from the boxes array.

        # Apply the mask to the image (overlay with a random color)
        # For visual differentiation, a random color is generated for each object. This color is applied to the region of
        # the image where the mask is True.
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # Random color for each object
        
        #Look at 'tutorial' on how below code is used
        # - The mask is overlaid on the original image by blending the color with the original pixel values. The formula below
        #   multiplies the original image pixel values by 0.5 and adds the corresponding color value (also multiplied by 0.5), 
        #   producing a semi-transparent overlay of the mask.
        # - Effect: This creates a soft blend between the original image and the mask, visually highlighting the region of the 
        #   object in the image.
        image_np[mask_np] = image_np[mask_np] * 0.5 + color * 0.5  # Blend the mask with the image

        # Draw bounding box
        # Draw Rectangle:
        # (int(box[0]), int(box[1])): The top-left corner of the box.
        # (int(box[2]), int(box[3])): The bottom-right corner of the box.
        # The bounding box is drawn with the same random color used for the mask, ensuring consistency between the highlighted 
        # mask and its bounding box.
        # The parameter 2 at the end of the function call specifies the thickness of the rectangle.
        cv2.rectangle(image_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color.tolist(), 2)
    
    # Display the result
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

# Step 4: Apply the segmentation model to an input image
image, prediction = segment_image(IMAGE_PATH)

# Step 5: Visualize the segmented objects in the image
visualize_segmentation(image, prediction)
