from ultralytics import YOLO
from pathlib import Path
import cv2 

# Path to the saved trained model
SAVED_MODEL_PATH = Path.cwd() / 'temp' / 'models' / 'yolov8n_trained.pt'
TEST_IMAGE_NAME = 'bisturi720.jpg'
TEST_IMAGE_PATH = Path.cwd() / 'temp' / 'data_preprocessed_step2' / 'test' / 'images' / TEST_IMAGE_NAME

# Path to save predictions
PREDICTIONS_DIR = Path.cwd() / 'predictions'
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTED_IMAGE_NAME = TEST_IMAGE_NAME[:-4] + "_pred.jpg"
ANNOTATED_IMAGE_PATH = PREDICTIONS_DIR / PREDICTED_IMAGE_NAME

# Load the trained model
model = YOLO(SAVED_MODEL_PATH)
print(f"Model loaded from {SAVED_MODEL_PATH}")

# Run inference on the test image
results_list = model(TEST_IMAGE_PATH)

# Since results_list is a list, access the first element
results = results_list[0]

# Access predictions programmatically
if results.boxes:  # Check if there are detected boxes
    for box in results.boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Coordinates: {box.xyxy}")
else:
    print("No objects detected.")

# Display results on the image
results.show()

# Save the annotated image using OpenCV
annotated_image = results.plot() # Plot the annotated image (returns a NumPy array)
cv2.imwrite(ANNOTATED_IMAGE_PATH, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
print(f"Annotated image saved to {ANNOTATED_IMAGE_PATH}")

# """
# Model loaded from C:\Users\hi\Desktop\projects\python_projects\ai_projects\computer_vision\crowd_object_detection\train_OnCustomDSImagesWithNClasses_YOLO8\temp\models\yolov8n_trained.pt

# image 1/1 C:\Users\hi\Desktop\projects\python_projects\ai_projects\computer_vision\crowd_object_detection\train_OnCustomDSImagesWithNClasses_YOLO8\temp\data_preprocessed_step2\test\images\bisturi166.jpg: 480x640 1 {'scalpel nº4': None}, 119.2ms
# Speed: 4.0ms preprocess, 119.2ms inference, 1.0ms postprocess per image at shape (1, 3, 480, 640)
# Class: tensor([2.]), Confidence: tensor([0.9830]), Coordinates: tensor([[305.8575, 163.3487, 467.8279, 336.9179]])
# Annotated image saved to C:\Users\hi\Desktop\projects\python_projects\ai_projects\computer_vision\crowd_object_detection\train_OnCustomDSImagesWithNClasses_YOLO8\predictions\bisturi166_pred.jpg
# """