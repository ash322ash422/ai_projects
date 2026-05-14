import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the video
video_path = '../test_waveform.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# Display the frame and select the ROI
roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Extract ROI coordinates
x, y, w, h = roi
print(f"ROI:{roi}")
# Process the video frame-by-frame
waveform_data = []

frame_count=0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Crop the ROI from the frame
    roi_frame = frame[y:y+h, x:x+w]

    # Convert to grayscale for analysis
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    
    # Optional: Apply thresholding to make the waveform more distinct
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # print(f"thresh: {thresh}")
    
    # Sum pixel values along each row (to get waveform intensity)
    waveform = np.sum(thresh, axis=0)
    waveform_data.append(waveform)
    print(f"frame count{frame_count}")
    frame_count +=1

# Release the video capture
cap.release()

# Convert waveform data to a numpy array
waveform_data = np.array(waveform_data)

# Plot the extracted waveform over time
plt.figure(figsize=(10, 6))
plt.imshow(waveform_data.T, aspect='auto', cmap='hot', extent=[0, waveform_data.shape[0], 0, waveform_data.shape[1]])
plt.xlabel('Frame Number')
plt.ylabel('Waveform Position')
plt.title('Extracted ECG Waveform')
plt.colorbar(label='Intensity')
plt.show()
