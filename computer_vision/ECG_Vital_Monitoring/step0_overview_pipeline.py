import cv2
from threading import Thread
from time import sleep

class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.ret = True
        self.frame = None
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True  # Ensure the thread stops when the main program exits
        self.thread.start()

    def update(self):
        while self.ret:
            self.ret, self.frame = self.cap.read()
            sleep(0.02)  # Small delay to avoid overwhelming the CPU

    def get_frame(self):
        return self.frame

    def release(self):
        self.cap.release()

def main():
    video_path = 'td.mp4'
    processor = VideoProcessor(video_path)

    if not processor.cap.isOpened():
        print("Error: Cannot open video file")
        return
    
    frame_count = 0
    while True:
        frame = processor.get_frame()
        if frame is None:
            continue  # Wait until the first frame is available
        
        # Processing logic here (ROI detection, OCR, etc.)
        
        frame_count += 1  # Increment the frame count
        
        # Display the frame count on the frame
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    processor.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
