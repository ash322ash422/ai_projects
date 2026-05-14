import cv2, uuid, easyocr,  torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
# IMAGE_PATH = Path.cwd() / 'images' / 'busy_street_scene_with_multiple_objects_like_cars_people_traffic_lights_buildings.jpg'
IMAGE_PATH = Path.cwd() / 'images' / 'street_scene_with_objects_like_signs_banners_that_contain_visible_text_in_English.jpg'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # You can add languages like ['en', 'fr'] for English and French

# Load the Faster R-CNN model for object detection
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load the CLIP model and processor for object identification
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

CLIP_LABELS = [
    "cat", "dog", "car", "person", "bicycle", "chair", "table", "tree", "flower", "computer",
    "phone", "book", "cup", "ball", "airplane", "fish", "elephant", "horse", "sheep", "cow",
    "bird", "motorcycle", "train", "bus", "truck", "pizza", "sandwich", "cookie", "donut", "cake"
]

def detect_and_extract_text(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    extracted_data = []

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
            text_inputs = clip_processor(text=["a photo of a " + label for label in CLIP_LABELS], return_tensors="pt", padding=True)
            image_inputs = clip_processor(images=Image.fromarray(detected_image), return_tensors="pt")
            
            # Compute similarity between image and text
            with torch.no_grad():
                image_features = clip_model.get_image_features(**image_inputs)
                text_features = clip_model.get_text_features(**text_inputs)
                logits_per_image = torch.matmul(image_features, text_features.T)

            # Get the most similar label
            probs = logits_per_image.softmax(dim=1)
            most_similar_label = CLIP_LABELS[torch.argmax(probs).item()]

            # Extract text using EasyOCR
            detected_image_bgr = cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR)  # EasyOCR expects BGR format
            extracted_text = reader.readtext(detected_image_bgr, detail=0)  # Extract text without detailed coordinates

            # Store object metadata
            object_info = {
                "object_id": object_id,
                "object_label": most_similar_label,
                "confidence": score,
                "extracted_text": extracted_text
            }
            extracted_data.append(object_info)
            print(f"Object ID: {object_id}, Label: {most_similar_label}, Extracted Text: {extracted_text}, confidence: {score}")
    
    return extracted_data

# Call the function with your image path
# Example: detect_and_extract_text("path_to_your_image.jpg")

if __name__ == "__main__":
    image_path = IMAGE_PATH  # Provide your image path here
    extracted_data = detect_and_extract_text(image_path)
    # print("extracted_data=\n",extracted_data)