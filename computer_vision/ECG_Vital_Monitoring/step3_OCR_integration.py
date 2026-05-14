import cv2
from threading import Thread
from time import sleep
import math
import pytesseract

# Set the path to tesseract executable (update this with your actual path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def initialize_video(video_path):
    """Initialize the video capture."""
    cap = cv2.VideoCapture(video_path)
    return cap

def _start_video_thread(cap, target_fps, frame_callback):
    """Start a thread to continuously read and process frames."""
    frame_interval = 1 / target_fps

    def update():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is not None:
                processed_frame = _preprocess_frame(frame)
                roi_frame = _detect_roi(processed_frame)
                frame_callback(roi_frame)

            sleep(frame_interval)

    thread = Thread(target=update, daemon=True)
    thread.start()
    return thread

def _preprocess_frame(frame):
    """Preprocess the frame with resizing, normalization, denoising, and grayscale conversion."""
    # 1. Resizing to a standard size (e.g., 640x480)
    frame = cv2.resize(frame, (640, 480))

    # 2. Normalization (scaling pixel values to [0, 1])
    frame = frame / 255.0

    # 3. Denoising (Gaussian Blur)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # 4. Color space conversion to grayscale
    frame = cv2.cvtColor((frame * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)

    return frame

def _detect_roi(frame):
    """Detect and highlight regions of interest (ROI) in the frame and perform OCR."""
    # 1. Edge Detection using Canny
    edges = cv2.Canny(frame, 50, 150)

    # 2. Contour Detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for visualization

    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]

            # Preprocess ROI for OCR (binarization and adaptive thresholding)
            roi_bin = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

            # Extract text using Tesseract OCR
            text = pytesseract.image_to_string(roi_bin, config='--psm 6 digits')

            # Draw bounding box and OCR text on the frame
            cv2.rectangle(roi_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(roi_frame, text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # 3. Template Matching or Hough Transform (Example: Detect lines in the waveform)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(roi_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return roi_frame

def display_video(cap, target_fps):
    """Main loop to display the processed video frames with frame count."""
    frame_count = 0
    latest_frame = [None]

    def frame_callback(frame):
        latest_frame[0] = frame

    # Start the video processing thread
    _start_video_thread(cap, target_fps, frame_callback)

    while True:
        frame = latest_frame[0]
        if frame is None:
            continue  # Wait until the first frame is available

        # Increment the frame count
        frame_count += 1

        # Display the frame count on the frame
        display_frame = cv2.putText(frame.copy(), f"Frame: {frame_count}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Frame", display_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = 'td.mp4'
    target_fps = 30

    cap = initialize_video(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    display_video(cap, target_fps)

if __name__ == "__main__":
    main()
