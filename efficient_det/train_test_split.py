import json
import os
from sklearn.model_selection import train_test_split

# 设置路径
dataset_path = "/Users/ryanlil86/Desktop/database/job/intern/字节跳动/EfficientDet"  # 数据集的根目录路径
coco_json = os.path.join(dataset_path, "output_coco_format.json")  # 原始COCO格式JSON文件
train_json = os.path.join(dataset_path, "train_annotations.json")  # 输出的训练集COCO格式JSON文件
val_json = os.path.join(dataset_path, "val_annotations.json")  # 输出的验证集COCO格式JSON文件

# 加载原始COCO格式的JSON文件
with open(coco_json, 'r') as f:
    data = json.load(f)

# 获取所有图像的ID列表
image_ids = [image['id'] for image in data['images']]

# 使用sklearn的train_test_split将数据集划分为训练集和验证集(80%训练, 20%验证)
train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

# 创建训练集和验证集的COCO格式数据结构
train_data = {
    "images": [],
    "annotations": [],
    "categories": data["categories"]
}

val_data = {
    "images": [],
    "annotations": [],
    "categories": data["categories"]
}

# 将图像和注释分配到训练集和验证集中
train_image_ids = set(train_ids)
val_image_ids = set(val_ids)

for image in data['images']:
    if (image['id'] in train_image_ids):
        train_data['images'].append(image)
    elif (image['id'] in val_image_ids):
        val_data['images'].append(image)

for annotation in data['annotations']:
    if (annotation['image_id'] in train_image_ids):
        train_data['annotations'].append(annotation)
    elif (annotation['image_id'] in val_image_ids):
        val_data['annotations'].append(annotation)

# 将训练集和验证集分别写入新的COCO格式JSON文件
with open(train_json, 'w') as f:
    json.dump(train_data, f, indent=4)

with open(val_json, 'w') as f:
    json.dump(val_data, f, indent=4)

print(f"Training annotations saved to: {train_json}")
print(f"Validation annotations saved to: {val_json}")
