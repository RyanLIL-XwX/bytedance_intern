import os  # 提供与操作系统交互的功能
import shutil  # 提供高级文件操作功能
import random  # 提供生成随机数的功能

# 数据集路径
dataset_path = "/Users/ryanlil86/Desktop/database/job/intern/字节跳动/EfficientDet/NEU surface defect database"
train_path = os.path.join(dataset_path, "train")  # 训练集路径
val_path = os.path.join(dataset_path, "val")  # 验证集路径

# 创建目标文件夹，如果它们不存在
os.makedirs(os.path.join(train_path, "images"), exist_ok=True)
os.makedirs(os.path.join(train_path, "labels"), exist_ok=True)
os.makedirs(os.path.join(val_path, "images"), exist_ok=True)
os.makedirs(os.path.join(val_path, "labels"), exist_ok=True)

# 获取所有图片文件（假设是.jpg和.bmp文件）
image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.bmp'))]

# 随机打乱文件列表
random.shuffle(image_files)

# 计算文件列表的划分点
split_index = int(len(image_files) * 0.8)

# 分配文件到训练集和验证集
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# 移动文件到对应的文件夹
for file_name in train_files:
    # 移动图片文件
    shutil.move(os.path.join(dataset_path, file_name), os.path.join(train_path, "images", file_name))
    # 移动对应的标签文件
    label_file = file_name.replace('.jpg', '.txt').replace('.bmp', '.txt')
    if os.path.exists(os.path.join(dataset_path, label_file)):
        shutil.move(os.path.join(dataset_path, label_file), os.path.join(train_path, "labels", label_file))

# 移动验证集文件
for file_name in val_files:
    # 移动图片文件
    shutil.move(os.path.join(dataset_path, file_name), os.path.join(val_path, "images", file_name))
    # 移动对应的标签文件
    label_file = file_name.replace('.jpg', '.txt').replace('.bmp', '.txt')
    if os.path.exists(os.path.join(dataset_path, label_file)):
        shutil.move(os.path.join(dataset_path, label_file), os.path.join(val_path, "labels", label_file))
