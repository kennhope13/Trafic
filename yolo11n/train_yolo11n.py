from ultralytics import YOLO
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())

model = YOLO("yolo11n.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=0
)
