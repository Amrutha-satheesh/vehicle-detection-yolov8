from ultralytics import YOLO
import cv2
import os

# load your trained model
model_path = r"C:\Users\DELL\Desktop\DESKTOP\Nextgenpro\DL\vechicleDetection_yolo\runs\detect\train-2\weights\best.pt"
model = YOLO("yolov8n.pt")
print("loaded trained yolov8 vechicle model")

# test image path
image_path = r"C:\Users\DELL\Desktop\DESKTOP\Nextgenpro\DL\vechicleDetection_yolo\valid\images\1d3c0507e127f1ea_jpg.rf.2c6a1e3b13c7582a2072e3204c8f010d.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"image not found:{image_path}")

# run detection
results = model.predict(
    source=image_path,
    conf=0.25,
    device="cpu"
)

# draw results
annotated_image = results[0].plot()

# show image
cv2.imshow("vechicle detection",annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()