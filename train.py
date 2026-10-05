from ultralytics import YOLO

model=YOLO("yolov8n.pt")
model.train(data="dataset.yaml",epochs=2,imgsz=640,device="cpu",workers=0)
