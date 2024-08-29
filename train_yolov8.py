# pip3 install ultralytics # 安装ultralytics库, YOLOv8的官方库
from ultralytics import YOLO

# 加载模型(可以是预训练模型)
"""
'yolov8n.pt'表示YOLOv8的nano版本模型, 这是一个较小的模型, 适用于计算资源有限或需要快速推理的场景.
YOLOv8还提供了其他版本的模型, 例如s(small)、m(medium)、l(large)和x(extra-large), 分别适用于不同的精度和速度要求.
"""
model = YOLO('yolov8n.pt')  # 'n' for nano model, replace with 's', 'm', 'l', 'x' for larger models

# 使用指定的数据集和训练参数开始训练模型
"""
data.yaml文件包含了训练和验证数据集的路径、类别数量和类别名称等信息. YOLOv8会根据这个文件加载数据集.
epochs: 指定训练的轮数. 更多的轮数通常意味着更好的模型收敛, 但也会增加训练时间.
imgsz: 指定输入图像的大小. YOLOv8将所有输入图像调整为640x640像素. 较大的图像尺寸通常会带来更好的检测效果，但会增加计算开销.
batch: 指定每个批次的图像数量. 批次大小会影响训练的内存需求和模型更新的频率. 较大的批次可能需要更多的GPU内存, 但可以使训练过程更加稳定.
"""
model.train(data='data.yaml', epochs=100, imgsz=640, batch=16)