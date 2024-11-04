import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd
from pathlib import Path
# IMAGE_PATH = Path.cwd() / 'images' / 'busy_street_scene_with_multiple_objects_like_cars_people_traffic_lights_buildings.jpg'
IMAGE_PATH = Path.cwd() / 'images' / 'busy_street_scene_with_multiple_objects_like_cars_people_traffic_lights_buildings.jpg'

# Step 1: Load the image (replace 'input_image.jpg' with the path to your image)
image_path = IMAGE_PATH
image = Image.open(image_path)

# Step 2: Mock data (for this example, we'll use some manually created bounding boxes and labels)
mapped_data = {
    "master_image_id": "5f35d803-2dbf-4c36-aaf9-c7b8a89db6d8",
    "objects": [
        {"object_id": "9f473276-55ef-4c12-98fa-5acbc1533457", "label": "person", "extracted_text": [], "summary": "A person standing near a car.", "bbox": [50, 50, 150, 200]},
        {"object_id": "63b859e5-14f2-4a1f-9fa5-bd874c6e6d3c", "label": "car", "extracted_text": ["BMW"], "summary": "A blue BMW car.", "bbox": [200, 150, 300, 250]},
        {"object_id": "6b473276-65ef-4c12-9f8a-7acbc2333459", "label": "street sign", "extracted_text": ["Stop", "Speed Limit 50"], "summary": "A stop sign and speed limit sign.", "bbox": [350, 50, 400, 100]},
    ]
}

# Step 3: Create the table summarizing the mapped data using pandas
data = []
for obj in mapped_data['objects']:
    data.append([obj['object_id'], obj['label'], ", ".join(obj['extracted_text']), obj['summary']])
    
df = pd.DataFrame(data, columns=['Object ID', 'Label', 'Extracted Text', 'Summary'])

# Step 4: Display the image with annotated bounding boxes and labels
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.imshow(image)

# Annotating each object in the image
for obj in mapped_data['objects']:
    bbox = obj['bbox']
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    # Adding label as text annotation
    ax.text(bbox[0], bbox[1] - 10, f"{obj['label']} ({obj['object_id'][:4]})", color='white', fontsize=10, backgroundcolor='black')

# Show the image with annotations
plt.axis('off')  # Turn off the axis
plt.show()

# Step 5: Output the table as a summary of the mapped data
print("\n--- Data Table Summary ---")
print(df)

# Optionally, save the table to a file
df.to_csv("object_data_summary.csv", index=False)
