from dotenv import load_dotenv
load_dotenv()

import argparse, os

import cv2
from inference.models.utils import get_roboflow_model

import supervision as sv

if __name__ == "__main__":
    
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if api_key is None:
        raise ValueError(
            "Roboflow API key is missing. Please provide it as an argument or set the "
            "ROBOFLOW_API_KEY environment variable."
        )
    
    model = get_roboflow_model("yolov8x-640")
    
    bounding_box_annotator = sv.BoxAnnotator(thickness=4)
    
    frame_generator = sv.get_video_frames_generator(source_path="data/vehicles.mp4")

    for frame in frame_generator:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        
        annotated_frame = frame.copy()
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        
        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cv2.destroyAllWindows() 
        
        