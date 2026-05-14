"""
Logic flow:
1. Initialize object detection model and tracking algorithm.
2. Load the video using OpenCV.
3. For each frame:
   a. Detect persons and extract bounding boxes.
   b. Calculate distances between detected persons.
   c. Identify clusters of persons in close proximity.
4. Track identified groups over consecutive frames:
   a. Record group persistence across frames.
   b. Check for groups persisting for 10+ frames.
5. If a crowd is detected:
   a. Log frame number and crowd size.
6. Save results as a CSV file.

"""
import cv2, os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib  import Path

DEBUG=False #For debugging and viewing frame by frame. Close the frame before moving forward

# File paths
VIDEO_PATH        = Path.cwd() / 'temp' / 'dataset_video.mp4'
OUTPUT_VIDEO_PATH = Path.cwd() / 'temp' / 'output_crowd_detection.avi'
OUTPUT_CSV        = Path.cwd() / 'crowd_detection_log.csv'
MODEL_PATH        = Path.cwd() / 'temp' / 'yolov8m.pt'

DISTANCE_THRESHOLD = 80 # Distance threshold (in pixels)
FRAME_PERSISTENCE_THRESHOLD = 5  # Number of consecutive frames a group must persist to be considered a crowd

# Initialize the YOLO model using pre-trained weights
model = YOLO(MODEL_PATH)  # YOLOv8 nano version (lightweight), yolov8m.pt for medium

def detect_persons(frame):
    """Detect persons in a frame"""
    results = model(frame)  # Perform detection
    persons = []
    
    # Handle batch or single-frame predictions
    if isinstance(results, list):  
        results = results[0]
    
    for box in results.boxes:  # Each box contains detection info
        # Access attributes individually
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class ID

        # Only consider persons (class 0 in COCO dataset)
        if cls == 0:
            persons.append((x1, y1, x2, y2))
    return persons

def calculate_distances(persons):
    """Calculate pairwise distances between detected persons."""
    centers = [(int((x1+x2)/2), int((y1+y2)/2)) for x1, y1, x2, y2 in persons]
    distances = np.zeros((len(centers), len(centers)))
    for i, (x1, y1) in enumerate(centers):
        for j, (x2, y2) in enumerate(centers):
            distances[i, j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distances, centers

def identify_crowds(persons, distances):
    """Identify groups of 3 or more persons in close proximity."""
    clusters = []
    for i, _ in enumerate(persons):
        group = [i]
        for j in range(len(persons)):
            if i != j and distances[i, j] < DISTANCE_THRESHOLD:
                group.append(j)
        if len(group) >= 3:  # Cluster with 3+ persons
            clusters.append(group)
    return clusters

def draw_visualization(frame, frame_number, persons, centers, clusters):
    """Draw bounding boxes and labels for detected persons and crowds."""
    for x1, y1, x2, y2 in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for persons
    
    if DEBUG: #For debugging
        cv2.imshow(f'Detection frame_number : {frame_number}', frame)
        cv2.waitKey(0)  # Close the frame before moving forward. press any key to continue

    for cluster in clusters:
        # Highlight crowd centers with a circle
        cluster_centers = [centers[i] for i in cluster]
        for cx, cy in cluster_centers:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red for crowd centers
        
        # Draw bounding box for crowd area
        cluster_coords = [persons[i] for i in cluster]
        x1 = min(p[0] for p in cluster_coords)
        y1 = min(p[1] for p in cluster_coords)
        x2 = max(p[2] for p in cluster_coords)
        y2 = max(p[3] for p in cluster_coords)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for crowds
        
        # IMPORTANT: Add crowd size label and show frame_number only if cluster exist
        cv2.putText(frame, f'Crowd: {len(cluster)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(
            frame, f'Frame: {frame_number}', (10, 30),  # Position (10, 30) for top-left corner
            cv2.FONT_HERSHEY_SIMPLEX, 1,  # Font type and scale
            (255, 255, 255), 2  # White text with thickness 2
        )

def main():
    
    # Video and output initialization
    video = cv2.VideoCapture(VIDEO_PATH)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    frame_number = 0
    crowd_log = []
    persistent_groups = {}

    while video.isOpened():
        print(f"\n\nAnalyzing frame number:{frame_number}")
        ret, frame = video.read()
        if not ret:
            break

        frame_number += 1
        persons = detect_persons(frame)
        if len(persons) < 3:
            out_video.write(frame)  # Write original frame if no crowds detected
            continue

        distances, centers = calculate_distances(persons)
        clusters = identify_crowds(persons, distances)
        print(f"....Frame {frame_number}:")
        print(f"distances.shape = {distances.shape}")
        print(f"distances\n = {distances}")
        print(f"centers =           {clusters}")
        print(f"Detected clusters = {clusters}")
        
        # Track groups over frames
        for cluster in clusters:
            cluster_id = tuple(sorted(cluster))
            if cluster_id in persistent_groups:
                persistent_groups[cluster_id]['frames'] += 1
            else:
                persistent_groups[cluster_id] = {'frames': 1, 'count': len(cluster)}

        # Check for persistent groups
        for group_id, data in list(persistent_groups.items()):
            if data['frames'] >= FRAME_PERSISTENCE_THRESHOLD:
                crowd_log.append({'Frame': frame_number, 'Person_Count': data['count']})
                del persistent_groups[group_id]

        # Add visualization
        draw_visualization(frame,frame_number, persons, centers, clusters) 
        out_video.write(frame)  # Write the annotated frame

    # Release video resources
    video.release()
    out_video.release()

    # Save crowd detection log to CSV
    if crowd_log:
        pd.DataFrame(crowd_log).to_csv(OUTPUT_CSV, index=False)
        print(f"CSV {OUTPUT_CSV} saved successfully.")
    else:
        print("No crowds detected. CSV not written.")
        
    print(f"Video file {OUTPUT_VIDEO_PATH} saved successfully.") 
    
    
if __name__ == '__main__':
    main()
    
"""
Analyzing frame number:0

0: 384x640 34 persons, 1 handbag, 450.3ms
Speed: 6.5ms preprocess, 450.3ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)
....Frame 1:
distances.shape = (34, 34)
distances
 = [[          0      1111.8      554.36 ...      1067.5      1041.4      965.52]
 [     1111.8           0      632.68 ...      1806.3      1750.9      1488.6]
 [     554.36      632.68           0 ...      1202.6      1152.1      927.95]
 ...
 [     1067.5      1806.3      1202.6 ...           0       64.07      378.38]
 [     1041.4      1750.9      1152.1 ...       64.07           0      314.31]
 [     965.52      1488.6      927.95 ...      378.38      314.31           0]]
centers =           [[25, 22, 28], [31, 26, 32]]
Detected clusters = [[25, 22, 28], [31, 26, 32]]


Analyzing frame number:1

0: 384x640 35 persons, 1 handbag, 910.6ms
Speed: 4.0ms preprocess, 910.6ms inference, 3.0ms postprocess per image at shape (1, 3, 384, 640)
....Frame 2:
distances.shape = (35, 35)
distances
 = [[          0      1108.2      634.72 ...      1827.2      1749.2      1531.8]
 [     1108.2           0      555.96 ...      1038.6      1037.8      926.07]
 [     634.72      555.96           0 ...      1208.7        1144      942.42]
 ...
 [     1827.2      1038.6      1208.7 ...           0      135.59      348.05]
 [     1749.2      1037.8        1144 ...      135.59           0      231.14]
 [     1531.8      926.07      942.42 ...      348.05      231.14           0]]
centers =           [[13, 20, 30], [25, 23, 26]]
Detected clusters = [[13, 20, 30], [25, 23, 26]]


Analyzing frame number:2

0: 384x640 35 persons, 1 handbag, 450.4ms
Speed: 4.0ms preprocess, 450.4ms inference, 3.0ms postprocess per image at shape (1, 3, 384, 640)
....Frame 3:
distances.shape = (35, 35)
distances
 = [[          0      1106.9      1370.6 ...      1528.9      1823.9      1719.7]
 [     1106.9           0      340.77 ...      919.26      1029.9      1022.4]
 [     1370.6      340.77           0 ...      682.98      714.22      737.99]
 ...
 [     1528.9      919.26      682.98 ...           0      349.38      199.68]
 [     1823.9      1029.9      714.22 ...      349.38           0      169.96]
 [     1719.7      1022.4      737.99 ...      199.68      169.96           0]]
centers =           [[13, 18, 28], [25, 24, 26], [29, 31, 34]]
Detected clusters = [[13, 18, 28], [25, 24, 26], [29, 31, 34]]


Analyzing frame number:3

0: 384x640 34 persons, 431.8ms
Speed: 3.0ms preprocess, 431.8ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)
....Frame 4:
distances.shape = (34, 34)
distances
 = [[          0      1105.4      640.26 ...      1805.4      1761.5      1721.2]
 [     1105.4           0      558.29 ...      1058.4      937.55      1020.9]
 [     640.26      558.29           0 ...      1184.9        1128      1107.5]
 ...
 [     1805.4      1058.4      1184.9 ...           0      157.69      99.247]
 [     1761.5      937.55        1128 ...      157.69           0       203.6]
 [     1721.2      1020.9      1107.5 ...      99.247       203.6           0]]
centers =           [[14, 19, 27], [25, 23, 26], [30, 31, 33]]
Detected clusters = [[14, 19, 27], [25, 23, 26], [30, 31, 33]]


Analyzing frame number:4

0: 384x640 35 persons, 1 backpack, 423.2ms
Speed: 4.0ms preprocess, 423.2ms inference, 1.5ms postprocess per image at shape (1, 3, 384, 640)
....Frame 5:
distances.shape = (35, 35)
distances

...ommitted...


Analyzing frame number:340

0: 384x640 28 persons, 3 handbags, 354.5ms
Speed: 2.0ms preprocess, 354.5ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)
....Frame 341:
distances.shape = (28, 28)
distances
..omitted...
centers =           [[13, 11, 25]]
Detected clusters = [[13, 11, 25]]


Analyzing frame number:341



"""