import torch
from torchvision import models, transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Directory for saving segmented regions
os.makedirs("extracted_seg_regions", exist_ok=True)

# Load a pre-trained Mask R-CNN model
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transform input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transform(image).unsqueeze(0), image

# Post-process model output
def process_output(output, original_image):
    masks = output[0]['masks']
    labels = output[0]['labels']
    scores = output[0]['scores']
    
    # Threshold for displaying objects
    threshold = 0.5
    mask_count = 0
    
    for i, score in enumerate(scores):
        if score > threshold:
            mask = masks[i, 0].detach().cpu().numpy()
            label = labels[i].item()
            
            # Extract masked region and save it
            mask_applied = cv2.bitwise_and(original_image, original_image, mask=(mask > 0.5).astype(np.uint8))
            mask_filename = f"extracted_seg_regions/seg_region_{mask_count}_label_{label}.png"
            cv2.imwrite(mask_filename, cv2.cvtColor(mask_applied, cv2.COLOR_RGB2BGR))
            mask_count += 1
            
            # Overlay mask on the original image
            original_image[mask > 0.5] = [0, 255, 0]  # Mark segmented region in green
            
    return original_image

# Main function
def segment_image(image_path):
    input_image, original_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_image)
    segmented_image = process_output(output, original_image)
    
    # Save and display visual output
    output_filename = "segmented_output.png"
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.savefig(output_filename)
    print(f"Visual output saved as {output_filename}")

# Example usage
image_path = "input_image.jpg"  # Replace with the path to your image
segment_image(image_path)
