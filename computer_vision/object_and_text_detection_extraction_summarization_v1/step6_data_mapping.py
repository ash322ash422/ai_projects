import json
import uuid

# Step 1: Mock data for the master image and extracted objects
master_image_id = str(uuid.uuid4())  # Unique ID for the master image

# Step 2: List of extracted objects with mock data
extracted_objects = [
    {"label": "person", "text": [], "summary": "A person standing near a car."},
    {"label": "car", "text": ["BMW"], "summary": "A blue BMW car."},
    {"label": "street sign", "text": ["Stop", "Speed Limit 50"], "summary": "A stop sign and speed limit sign."},
    {"label": "shop sign", "text": ["Fresh Mart"], "summary": "A grocery store sign reading 'Fresh Mart'."},
]

# Step 3: Generate a mapping structure with unique IDs for each object
mapped_data = {
    "master_image_id": master_image_id,
    "objects": []
}

for obj in extracted_objects:
    # Each object gets a unique ID
    object_id = str(uuid.uuid4())
    
    # Map the object attributes: unique ID, label, extracted text, and summary
    mapped_data["objects"].append({
        "object_id": object_id,
        "label": obj["label"],
        "extracted_text": obj["text"],
        "summary": obj["summary"]
    })

# Step 4: Store the mapped data into a JSON structure
mapped_json = json.dumps(mapped_data, indent=4)
print(mapped_json)

# Step 5: Optionally, save the JSON data to a file
with open("object_data_mapping.json", "w") as json_file:
    json_file.write(mapped_json)

"""
{
    "master_image_id": "0fb4397d-6b01-45a7-9107-34240dba7da7",
    "objects": [
        {
            "object_id": "9ccb9e8d-3b7a-4c01-8f5f-6ce5c6a57e58",
            "label": "person",
            "extracted_text": [],
            "summary": "A person standing near a car."
        },
        {
            "object_id": "f1c3ccb5-86a8-4caa-a63f-5a0692c7cbbe",
            "label": "car",
            "extracted_text": [
                "BMW"
            ],
            "summary": "A blue BMW car."
        },
        {
            "object_id": "1a7c7dd5-59d5-4aa0-b9fc-8ad91dbf1d1c",
            "label": "street sign",
            "extracted_text": [
                "Stop",
                "Speed Limit 50"
            ],
            "summary": "A stop sign and speed limit sign."
        },
        {
            "object_id": "5e0368d2-291b-4451-9f51-1f13523d152a",
            "label": "shop sign",
            "extracted_text": [
                "Fresh Mart"
            ],
            "summary": "A grocery store sign reading 'Fresh Mart'."
        }
    ]
}
"""