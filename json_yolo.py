import json
import os

# 加载COCO格式的JSON文件
with open('output_coco_format.json') as f:
    data = json.load(f)

# 创建保存YOLO格式标注的文件夹
os.makedirs('labels', exist_ok=True)

for annotation in data['annotations']:
    image_id = annotation['image_id']
    image_info = next(item for item in data['images'] if item['id'] == image_id)
    
    # 计算中心点坐标和宽高，归一化
    if annotation['bbox']:
        x, y, w, h = annotation['bbox']
        x_center = (x + w / 2) / image_info['width']
        y_center = (y + h / 2) / image_info['height']
        w = w / image_info['width']
        h = h / image_info['height']
        
        # 保存为YOLO格式
        yolo_format = f"{annotation['category_id']-1} {x_center} {y_center} {w} {h}"
        label_path = os.path.join('labels', f"{image_info['file_name'].replace('.jpg', '.txt')}")
        
        with open(label_path, 'a') as label_file:
            label_file.write(yolo_format + '\n')
