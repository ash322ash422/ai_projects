import torch
from transformers import pipeline
from PIL import Image
from pathlib import Path
# IMAGE_PATH = Path.cwd() / 'images' / 'busy_street_scene_with_multiple_objects_like_cars_people_traffic_lights_buildings.jpg'
IMAGE_PATH = Path.cwd() / 'images' / 'street_scene_with_objects_like_signs_banners_that_contain_visible_text_in_English.jpg'

# Initialize an object detection model (Faster R-CNN in this example)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 as an example

# Load an image
image_path = IMAGE_PATH
image = Image.open(image_path)

# Step 1: Detect objects
results = model(image)

# Step 2: Extract object attributes (e.g., bounding box, label)
objects = results.pandas().xyxy  # Dataframe containing bounding boxes and labels

# Step 3: Summarization using an NLP model
summarizer = pipeline("summarization")

summarized_objects = []
for idx, obj in enumerate(objects):  # Iterate over list using enumerate()
    obj_id = obj['class']
    label = obj['name']
    size = (obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin'])
    color = "color attributes here"  # Placeholder for extracting the color

    # Create a descriptive summary
    description = f"Object {idx}: This is a {label} located in the image with a size of {size}. Color information: {color}."
    summary = summarizer(description)[0]['summary_text']  # Use the summarizer model
    
    summarized_objects.append({
        'Object ID': obj_id,
        'Label': label,
        'Summary': summary
    })

# Step 4: Save or print summaries
for obj_summary in summarized_objects:
    print(f"Object ID:\n {obj_summary['Object ID']}, Summary: {obj_summary['Summary']}")

# Optionally save summaries to a file or database
