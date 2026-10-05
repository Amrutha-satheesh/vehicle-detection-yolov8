# 🚗 Vehicle Detection using YOLOv8

An end-to-end Computer Vision project that utilizes the Ultralytics **YOLOv8** framework to train and infer real-time vehicle detection on custom dataset splits.

---

## 📌 Key Features

- **Real-Time Object Detection:** Fine-tuned YOLOv8 architecture optimized for rapid and accurate bounding-box predictions.
- **Custom Dataset Pipeline:** Structured YAML configuration routing `train`, `test`, and `valid` image/label splits.
- **Automated Training & Evaluation:** Complete pipeline for model training, validation metrics logging, and inference generation.

---

## 📁 Repository Structure

```text
├── train/                  # Training image & label dataset split
├── test/                   # Testing image & label dataset split
├── valid/                  # Validation image & label dataset split
├── dataset.yaml            # Dataset configuration file with class paths & labels
├── train.py                # Python script to fine-tune YOLOv8 model
├── test.py                 # Python script for running inference/evaluations
└── README.md               # Project documentation
