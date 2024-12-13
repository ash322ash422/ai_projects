import cv2
import pandas as pd
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from transformers import CLIPProcessor, CLIPModel
import uuid
from PIL import Image
from pathlib import Path

# Directory containing the images
EXTRACTED_IMAGE_DIR = Path.cwd() / 'extracted_objects'
OUTPUT_CSV_FILE = "identified_objects.csv"

CLIP_LABELS = [
    "cat", "dog", "car", "person", "bicycle", "chair", "table", "tree", "flower", "computer",
    "phone", "book", "cup", "ball", "airplane", "fish", "elephant", "horse", "sheep", "cow",
    "bird", "motorcycle", "train", "bus", "truck", "pizza", "sandwich", "cookie", "donut", "cake"
]

# Load the Faster R-CNN model with pre-trained weights
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Perform object detection and identification
def detect_and_identify_objects(image_path):
    # Read the input image
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    identified_objects = []

    # Process predictions
    for i in range(len(predictions['boxes'])):
        box = predictions['boxes'][i].cpu().numpy()
        score = predictions['scores'][i].cpu().numpy()
        
        # Only consider predictions above a certain confidence threshold
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            detected_image = image_rgb[y1:y2, x1:x2]
            object_id = str(uuid.uuid4())

            # Identify object using CLIP
            text_inputs = clip_processor(
                text=["a photo of a " + label for label in CLIP_LABELS],
                return_tensors="pt",
                padding=True
            )
            image_inputs = clip_processor(images=Image.fromarray(detected_image), return_tensors="pt")

            # Compute similarity between image and text
            with torch.no_grad():
                image_features = clip_model.get_image_features(**image_inputs)
                text_features = clip_model.get_text_features(**text_inputs)

                # Compute the similarity scores
                logits_per_image = torch.matmul(image_features, text_features.T)

            # Get the most similar label
            probs = logits_per_image.softmax(dim=1)
            most_similar_label = CLIP_LABELS[torch.argmax(probs).item()]
            print(f"Detected Object: {most_similar_label}, Confidence: {score}")

            # Save object information
            identified_objects.append((object_id, str(image_path), most_similar_label, score))

    return identified_objects

def save_identified_objects(all_identified_objects):
    # Create a DataFrame to store identified objects
    df = pd.DataFrame(all_identified_objects, columns=["Object ID", "Image Path", "Description", "Confidence"])
    
    # Save DataFrame to a CSV file
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Identified objects saved to {OUTPUT_CSV_FILE}")

if __name__ == "__main__":
    all_identified_objects = []

    # Iterate through all image files in the directory
    for image_file in EXTRACTED_IMAGE_DIR.iterdir():
        if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:  # Filter for image files
            print(f"Processing image: {image_file}")
            identified_objects = detect_and_identify_objects(image_file)
            all_identified_objects.extend(identified_objects)

    # Save all identified objects to CSV
    save_identified_objects(all_identified_objects)
