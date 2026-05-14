import cv2
from threading import Thread
from time import sleep

class VideoProcessor:
    def __init__(self, video_path, target_fps=30):
        self.cap = cv2.VideoCapture(video_path)
        self.ret = True
        self.frame = None
        self.target_fps = target_fps
        self.frame_interval = 1 / target_fps  # Time interval between frames
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True  # Ensure the thread stops when the main program exits
        self.thread.start()

    def update(self):
        while self.ret:
            self.ret, frame = self.cap.read()
            if not self.ret:
                break

            # Preprocessing steps
            if frame is not None:
                frame = self.preprocess_frame(frame)

            self.frame = frame
            sleep(self.frame_interval)  # Control frame rate to match target FPS

    def preprocess_frame(self, frame):
        # 1. Resizing to a standard size (e.g., 640x480)
        frame = cv2.resize(frame, (640, 480))

        # 2. Normalization (scaling pixel values to [0, 1])
        frame = frame / 255.0

        # 3. Denoising (Gaussian Blur)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # 4. Color space conversion to grayscale
        frame = cv2.cvtColor((frame * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)

        return frame

    def get_frame(self):
        return self.frame

    def release(self):
        self.cap.release()

def main():
    video_path = 'td.mp4'
    processor = VideoProcessor(video_path, target_fps=30)

    if not processor.cap.isOpened():
        print("Error: Cannot open video file")
        return

    frame_count = 0
    while True:
        frame = processor.get_frame()
        if frame is None:
            continue  # Wait until the first frame is available

        # Increment the frame count
        frame_count += 1

        # Display the frame count on the frame
        display_frame = cv2.putText(frame.copy(), f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Frame", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    processor.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
