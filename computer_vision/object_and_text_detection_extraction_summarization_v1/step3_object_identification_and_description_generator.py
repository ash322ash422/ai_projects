import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from transformers import CLIPProcessor, CLIPModel
import os
import uuid
from PIL import Image  # Fix: Importing Image from PIL for image processing
from pathlib import Path
IMAGE_PATH = Path.cwd() / 'images' / 'busy_street_scene_with_multiple_objects_like_cars_people_traffic_lights_buildings.jpg'

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
    image = cv2.imread(image_path)
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
            identified_objects.append((object_id, detected_image))

            # Identify object using CLIP
            # Process the image and text inputs separately
            text_inputs = clip_processor(text=["a photo of a " + label for label in CLIP_LABELS], return_tensors="pt", padding=True)
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
            identified_objects[-1] += (most_similar_label, score)

    return identified_objects

def save_identified_objects(identified_objects):
    # Create a DataFrame to store identified objects
    df = pd.DataFrame(identified_objects, columns=["Object ID", "Image", "Description", "Confidence"])
    
    # Save DataFrame to a CSV file
    df.to_csv('identified_objects.csv', index=False)

    print("Identified objects saved to identified_objects.csv")
    

if __name__ == "__main__":
    image_path = IMAGE_PATH  # Provide your image path here
    identified_objects = detect_and_identify_objects(image_path)
    save_identified_objects(identified_objects)


# serWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1`. You can also use `weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# 2024-10-05 14:56:08.052098: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2024-10-05 14:56:09.455447: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# C:\Users\hi\Desktop\projects\python_projects\ai_projects\computer_vision\.venv\lib\site-packages\transformers\tokenization_utils_base.py:1617: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be deprecated in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
#   warnings.warn(
# Detected Object: car, Confidence: 0.9985961318016052
# Detected Object: car, Confidence: 0.9968200922012329
# Detected Object: car, Confidence: 0.9942504167556763
# Detected Object: car, Confidence: 0.9938839077949524
# Detected Object: car, Confidence: 0.9923535585403442
# Detected Object: bus, Confidence: 0.9916231036186218
# Detected Object: car, Confidence: 0.9913733601570129
# Detected Object: car, Confidence: 0.9881018400192261
# Detected Object: person, Confidence: 0.9874334931373596
# Detected Object: car, Confidence: 0.9838746786117554
# Detected Object: car, Confidence: 0.9770530462265015
# Detected Object: car, Confidence: 0.9714344143867493
# Detected Object: person, Confidence: 0.9712864756584167
# Detected Object: person, Confidence: 0.9679343700408936
# Detected Object: person, Confidence: 0.9652752876281738
# Detected Object: bus, Confidence: 0.9644063711166382
# Detected Object: person, Confidence: 0.9638804197311401
# Detected Object: person, Confidence: 0.9619070887565613
# Detected Object: person, Confidence: 0.9559411406517029
# Detected Object: person, Confidence: 0.9532865285873413
# Detected Object: car, Confidence: 0.9525784850120544
# Detected Object: person, Confidence: 0.9502628445625305
# Detected Object: car, Confidence: 0.9491754770278931
# Detected Object: car, Confidence: 0.946010410785675
# Detected Object: car, Confidence: 0.9395020604133606
# Detected Object: person, Confidence: 0.9394145011901855
# Detected Object: person, Confidence: 0.9375528693199158
# Detected Object: person, Confidence: 0.9345154762268066
# Detected Object: person, Confidence: 0.933013916015625
# Detected Object: person, Confidence: 0.9287410378456116
# Detected Object: person, Confidence: 0.9235181212425232
# Detected Object: person, Confidence: 0.9223343729972839
# Detected Object: person, Confidence: 0.9211148023605347
# Detected Object: person, Confidence: 0.9196374416351318
# Detected Object: person, Confidence: 0.9175397157669067
# Detected Object: car, Confidence: 0.9109132885932922
# Detected Object: person, Confidence: 0.908728301525116
# Detected Object: person, Confidence: 0.9054769277572632
# Detected Object: car, Confidence: 0.8922780752182007
# Detected Object: car, Confidence: 0.8901330232620239
# Detected Object: car, Confidence: 0.8761093616485596
# Detected Object: computer, Confidence: 0.8572521805763245
# Detected Object: person, Confidence: 0.806964635848999
# Detected Object: person, Confidence: 0.7689167261123657
# Detected Object: phone, Confidence: 0.7657169103622437
# Detected Object: car, Confidence: 0.7606998085975647
# Detected Object: person, Confidence: 0.7530282735824585
# Detected Object: phone, Confidence: 0.7361471652984619
# Detected Object: person, Confidence: 0.6887727379798889
# Detected Object: person, Confidence: 0.6776502728462219
# Detected Object: person, Confidence: 0.6718071699142456
# Detected Object: person, Confidence: 0.6297286152839661
# Detected Object: phone, Confidence: 0.5969759225845337
# Detected Object: phone, Confidence: 0.5815110206604004
# Detected Object: person, Confidence: 0.569098949432373
# Detected Object: train, Confidence: 0.5622529983520508
# Detected Object: person, Confidence: 0.5613009333610535
# Detected Object: person, Confidence: 0.5283012986183167
# Detected Object: person, Confidence: 0.5276020169258118
# Detected Object: person, Confidence: 0.5219544768333435
# Detected Object: car, Confidence: 0.517654538154602
# Identified objects saved to identified_objects.csv
