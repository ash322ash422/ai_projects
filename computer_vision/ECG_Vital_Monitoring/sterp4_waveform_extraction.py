import cv2
from threading import Thread
from time import sleep
import pytesseract
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
import neurokit2 as nk

# Set the path to Tesseract executable (update this with your actual path)
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
    frame = cv2.resize(frame, (640, 480))  # Resize to standard size
    frame = frame / 255.0  # Normalize
    frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Denoise with Gaussian Blur
    frame = cv2.cvtColor((frame * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return frame


def _detect_roi(frame):
    """Detect and highlight regions of interest (ROI) in the frame, perform OCR, and analyze ECG waveform."""
    edges = cv2.Canny(frame, 50, 150)  # Edge detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y + h, x:x + w]

            # Preprocess ROI for OCR
            roi_bin = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(roi_bin, config='--psm 6 digits')

            # Draw bounding box and OCR text on the frame
            cv2.rectangle(roi_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(roi_frame, text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Analyze ECG waveform if the ROI resembles a waveform
            waveform = _extract_waveform(roi)
            if waveform is not None:
                r_peaks = _analyze_ecg(waveform)
                if len(r_peaks) > 0:
                    _plot_r_peaks(waveform, r_peaks)

    return roi_frame


def _extract_waveform(roi):
    """Extract the waveform from the ROI and return as a 1D signal."""
    try:
        height, width = roi.shape
        if width < 60:  # Ensure the ROI width is sufficient for waveform extraction
            print("Warning: ROI too small to extract waveform")
            return None

        # Assume the waveform is the middle horizontal strip of the ROI
        roi_strip = roi[height // 2 - 10:height // 2 + 10, :]
        waveform = np.mean(roi_strip, axis=0)  # Average along the vertical axis to get 1D signal

        # Ensure the waveform length is at least 60
        if len(waveform) < 60:
            print("Warning: Extracted waveform too short, padding with zeros")
            waveform = np.pad(waveform, (0, 60 - len(waveform)), mode='constant')

        return waveform
    except Exception as e:
        print(f"Error extracting waveform: {e}")
        return None


def lowpass_filter(signal, cutoff=10, fs=1000, order=4):
    """Apply low-pass filter with dynamic pad length handling."""
    if len(signal) < 20:
        print("Warning: Signal too short for filtering")
        return signal
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal, padlen=min(len(signal) - 1, 20))


def _analyze_ecg(signal):
    """Analyze the ECG signal to identify R-peaks using NeuroKit2."""
    try:
        # Check if the signal is long enough for processing
        if len(signal) < 60:
            print("Error: Signal too short for ECG analysis")
            return []

        # Preprocess the signal (filtering)
        cleaned_signal = nk.ecg_clean(signal, sampling_rate=1000)

        # Find R-peaks
        _, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=1000)
        return rpeaks['ECG_R_Peaks']
    except Exception as e:
        print(f"Error analyzing ECG: {e}")
        return []
    
    
def _plot_r_peaks(signal, r_peaks):
    """Plot the ECG signal with R-peaks marked."""
    def plot():
        plt.figure(figsize=(10, 4))
        plt.plot(signal, label='ECG Signal')
        plt.scatter(r_peaks, signal[r_peaks], color='red', marker='o', label='R-peaks')
        plt.title('ECG Signal with R-peaks')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    # Ensure plotting runs on the main thread
    from threading import main_thread
    if main_thread().is_alive():
        plot()
    else:
        print("Warning: Cannot plot outside of the main thread")


def display_video(cap, target_fps):
    """Main loop to display the processed video frames with frame count."""
    frame_count = 0
    latest_frame = [None]

    def frame_callback(frame):
        latest_frame[0] = frame

    _start_video_thread(cap, target_fps, frame_callback)

    while True:
        frame = latest_frame[0]
        if frame is None:
            continue

        frame_count += 1
        display_frame = cv2.putText(frame.copy(), f"Frame: {frame_count}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Frame", display_frame)

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
