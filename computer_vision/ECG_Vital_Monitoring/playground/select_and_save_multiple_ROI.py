"""
Webcam Capture:
Run the script with a webcam connected. The code will open a window showing the live feed.

Selecting ROI:

Click and drag in the window to select an ROI rectangle.
The ROI is processed and added to the GIF each frame after selection. This way you can select multiple ROI's.
Press q to stop capturing and save the GIF.
"""

import cv2
import numpy as np
from PIL import Image

# Define the callback function for mouse events
def select_roi(event, x, y, flags, param):
    global roi_points, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        roi_selected = False
    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        roi_selected = True

# Initialize the video capture object (0 for webcam or use a video file path)
cap = cv2.VideoCapture(0)

# Create a window to display the video and set the callback function
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", select_roi)

# Initialize variables
roi_points = []
roi_selected = False
gif_frames = [] #holds multiple frmaes

while True:
    # Capture a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        break

    # Draw the ROI rectangle if it has been selected
    if roi_selected and len(roi_points) == 2:
        roi = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(roi, roi_points[0], roi_points[1], (255, 255, 255), -1)
        roi_frame = cv2.bitwise_and(frame, frame, mask=roi)
        gif_frames.append(Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)))
        cv2.rectangle(frame, roi_points[0], roi_points[1], (0, 255, 0), 2)  # Visual feedback on ROI

    # Display the frame in the window
    cv2.imshow("Video", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy the window
cap.release()
cv2.destroyAllWindows()

# Save the GIF if frames were captured
if gif_frames:
    gif_frames[0].save("roi.gif", save_all=True, append_images=gif_frames[1:], duration=100, loop=0)
    print("GIF saved as roi.gif")
else:
    print("No frames captured for the GIF.")
