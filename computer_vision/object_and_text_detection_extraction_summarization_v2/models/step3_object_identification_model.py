import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path

base_dir = Path.cwd().parent / "data"
extracted_object_dir = base_dir / "output" / "extracted_objects" #input
object_descriptions_file       = base_dir / "output" / "object_descriptions.csv" #output
    
# Ensure all images are converted to RGB
def load_image(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':  # Check if the image has 3 channels
        image = image.convert('RGB')  # Convert to RGB
    return np.array(image)

# Identify objects and generate descriptions
def identify_objects(extracted_object_dir,object_descriptions_file):
    # Load pre-trained YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

    object_descriptions = []
    
    for obj_image in sorted(os.listdir(extracted_object_dir)):
        image_path = os.path.join(extracted_object_dir, obj_image)
        try:
            image = load_image(image_path)  # Load and preprocess image
            results = model(image)  # Perform object detection
            predictions = results.pandas().xyxy[0]  # Extract predictions as a DataFrame
            
            # Generate description
            descriptions = []
            for _, row in predictions.iterrows():
                label = row['name']
                confidence = row['confidence']
                descriptions.append(f"{label} ({confidence:.2f})")
            
            # Combine descriptions
            description_text = "; ".join(descriptions) if descriptions else "Unknown"
            object_descriptions.append((obj_image, description_text))
        except Exception as e:
            object_descriptions.append((obj_image, f"Error processing image: {e}"))
    
    # Save descriptions to a CSV file
    df = pd.DataFrame(object_descriptions, columns=["Image Name", "Description"])
    df.to_csv(object_descriptions_file, index=False)
    
    return object_descriptions_file, object_descriptions

def main():
    out_file, _ = identify_objects(
        extracted_object_dir=extracted_object_dir, #input
        object_descriptions_file=object_descriptions_file #output
    )
    print(f"--Descriptions saved in {out_file}")
    
if __name__ == "__main__":
    main()