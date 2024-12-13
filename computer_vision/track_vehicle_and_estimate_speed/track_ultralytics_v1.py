from dotenv import load_dotenv
load_dotenv()

import argparse, os

import cv2
from ultralytics import YOLO # different

import supervision as sv

if __name__ == "__main__":
    video_path="data/vehicles.mp4"
    
    model = YOLO("yolov8x.pt") # different
    
    bounding_box_annotator = sv.BoxAnnotator(thickness=4)
    
    frame_generator = sv.get_video_frames_generator(source_path=video_path)

    for frame in frame_generator:
        results = model(frame)[0] # different
        detections = sv.Detections.from_ultralytics(results)
        
        annotated_frame = frame.copy()
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        
        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cv2.destroyAllWindows() 
        
        