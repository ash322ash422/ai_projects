from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")

# Train the model
results = model.train(data="temp/senam_yoga_dataset11/data.yaml", epochs=2, imgsz=640)

