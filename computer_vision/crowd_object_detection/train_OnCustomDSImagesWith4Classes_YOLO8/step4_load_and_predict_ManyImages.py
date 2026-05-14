from ultralytics import YOLO
from pathlib import Path
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Paths
SAVED_MODEL_PATH = Path.cwd() / 'temp' / 'models' / 'yolov8n_trained_run2.pt' #TODO yolov8n_trained_run1.ptun1 and yolov8n_trained_run2.pt
TEST_IMAGES_DIR  = Path.cwd() / 'temp' / 'data_preprocessed_step2' / 'test' / 'images'
TEST_LABELS_DIR =  Path.cwd() / 'temp' / 'data_preprocessed_step2' / 'test'  / 'labels'
OUTPUT_CSV = Path.cwd() / 'predictions.csv' #TODO run1 and run2

# Load the trained model
model = YOLO(SAVED_MODEL_PATH)
print(f"Model loaded from {SAVED_MODEL_PATH}")

# Prepare for evaluation
rows = []  # To store CSV rows
actual_labels = []
predicted_labels = []

# Iterate over all test images
image_files = list(TEST_IMAGES_DIR.glob("*.jpg"))  # Adjust extension if needed
for image_file in image_files:
    # Get corresponding label file
    label_file = TEST_LABELS_DIR / f"{image_file.stem}.txt"

    # Load ground truth labels
    if not label_file.exists():
        print(f"Warning: No label file for {image_file.name}, skipping.")
        continue
    with open(label_file, 'r') as f:
        ground_truth = [int(line.split()[0]) for line in f.readlines()]  # Extract class IDs

    # Run inference
    results = model(image_file)[0]

    # Extract predictions
    if results.boxes:
        for box in results.boxes:
            predicted_class = int(box.cls[0].item())  # Predicted class ID
            confidence = float(box.conf[0].item())  # Confidence score

            # Check if prediction matches any ground truth label
            for actual_label in ground_truth:
                match = 1 if predicted_class == actual_label else 0

                # Save to rows and lists
                rows.append([image_file.name, actual_label, predicted_class, confidence, match])
                actual_labels.append(actual_label)
                predicted_labels.append(predicted_class)
    else:
        print(f"No predictions for {image_file.name}")

# Save predictions to CSV
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image Name", "Actual Label", "Predicted Label", "Confidence Score", "Match"])
    writer.writerows(rows)
print(f"Results saved to {OUTPUT_CSV}")

# Generate and plot confusion matrix
conf_matrix = confusion_matrix(actual_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # Save confusion matrix as image
plt.show()

