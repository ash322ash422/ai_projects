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
            "Roboflow API key is missing. Please set the "
            "ROBOFLOW_API_KEY environment variable."
        )
    video_info = sv.VideoInfo.from_video_path(video_path="data/vehicles.mp4")
    model = get_roboflow_model("yolov8x-640")
    
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    
    frame_generator = sv.get_video_frames_generator(source_path="data/vehicles.mp4")

    for frame in frame_generator:
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        
        annotated_frame = frame.copy()
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        
        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cv2.destroyAllWindows() 
        
        