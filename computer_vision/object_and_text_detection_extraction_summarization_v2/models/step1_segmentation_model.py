import torch
from torchvision import models, transforms
import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import shutil
from conf import EXTRACTED_SEG_REGIONS
from pathlib import Path

base_dir = Path.cwd().parent / "data"
image_path            = base_dir / "input_images" / "input_image.jpg"
segmented_file = base_dir / "output" / "segmented_output.png"
extracted_seg_regions = base_dir / "output" / EXTRACTED_SEG_REGIONS

# Transform input image
def _preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transform(image).unsqueeze(0), image

# Post-process model output
def _post_process_output(output, original_image, extracted_seg_regions):
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
            mask_filename = os.path.join(extracted_seg_regions,f"seg_region_{mask_count}_label_{label}.png")
            cv2.imwrite(mask_filename, cv2.cvtColor(mask_applied, cv2.COLOR_RGB2BGR))
            mask_count += 1
            
            # Overlay mask on the original image
            original_image[mask > 0.5] = [0, 255, 0]  # Mark segmented region in green
    print(f"--Segmented region saved in directory {extracted_seg_regions}")        
    return original_image

# Main function
def segment_image(image_path,segmented_file, extracted_seg_regions):

    shutil.rmtree(extracted_seg_regions, ignore_errors=True)
    os.makedirs(extracted_seg_regions)
    
    # Load a pre-trained Mask R-CNN model
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    input_image, original_image = _preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_image)
    segmented_image = _post_process_output(output, original_image,extracted_seg_regions)
    
    # Save and display visual output
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.savefig(segmented_file)
    
    return segmented_file , extracted_seg_regions
    
def main():
    
    segmented_filename, extracted_seg_regions_dir = segment_image(
        image_path=image_path, #input
        segmented_file=segmented_file, #output
        extracted_seg_regions=extracted_seg_regions #output
    )
    print(f"--visual segmented_filename saved in  : {segmented_filename}")
    print(f"--extracted_seg_regions_dir: {extracted_seg_regions_dir}")
    
if __name__ == "__main__":
    main()