from ultralytics import YOLO
# 1. Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt') 
# 2. Train the model using our custom dataset
# This does the exact same thing as the terminal command we tried!
results = model.train(data='data.yaml', epochs=10, imgsz=640)