# pip3 install ultralytics # 安装ultralytics库, YOLOv8的官方库
from ultralytics import YOLO

# 加载模型（可以是预训练模型）
model = YOLO('yolov8n.pt')  # 'n' for nano model, replace with 's', 'm', 'l', 'x' for larger models

# 开始训练
model.train(data='data.yaml', epochs=100, imgsz=640, batch=16)