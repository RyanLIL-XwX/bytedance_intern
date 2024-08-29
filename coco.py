import os # 用于处理文件和目录的路径操作
import cv2 # OpenCV库的一部分, 用于处理图像的操作
import json # 用于处理JSON文件的读写操作
from PIL import Image # 用于打开和处理图像文件
from pycocotools.coco import COCO # 用于处理COCO数据集格式
from pycocotools.cocoeval import COCOeval # 用于评估COCO数据集格式

# 接收三个参数: 图像文件夹路径image_folder, 输出JSON文件路径output_json, 以及COCO格式中的类别信息categories
def convert_to_coco(image_folder, output_json, categories):
    # 初始化一个字典coco_format, 该字典将用于存储COCO格式的数据结构
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    annotation_id = 1 # 用于在COCO格式中唯一标识每个注释
    image_id = 1 # 用于在COCO格式中唯一标识每个图像

    # 遍历image_folder中的所有文件
    for filename in os.listdir(image_folder):
        # 确认是否是BMP文件
        if (filename.endswith(".bmp") == True):
            image_path = os.path.join(image_folder, filename) # 生成图像文件的完整路径
            img = Image.open(image_path) # 使用PIL的Image.open函数打开BMP图像文件
            img = img.convert("RGB")  # 转换成RGB格式
            jpeg_filename = filename.replace(".bmp", ".jpg") # 将BMP文件名中的扩展名.bmp替换为.jpg, 生成JPEG文件名
            jpeg_path = os.path.join(image_folder, jpeg_filename) # 生成JPEG文件的完整路径
            img.save(jpeg_path) # 将图像保存为JPEG格式文件
            # 存储图像的文件名, 图像高度, 宽度和图像ID
            image_info = {
                "file_name": jpeg_filename,
                "height": img.height,
                "width": img.width,
                "id": image_id
            }
            coco_format["images"].append(image_info) # 将图像信息添加到coco_format的images列表中

            # 存储注释信息: 这里的注释信息是示例内容, 包含了图像ID, 类别ID, 以及空的边界框(bbox)和分割信息
            annotation = {
                "segmentation": [],
                "area": 0,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [],
                "category_id": 1,  # 根据categories中的类别信息, 这里的类别ID为1
                "id": annotation_id
            }
            coco_format["annotations"].append(annotation) # 将注释信息添加到coco_format的annotations列表中

            annotation_id += 1
            image_id += 1

    # 打开或创建指定路径的JSON文件output_json, 以写入模式
    with open(output_json, 'w') as json_file:
        # 使用json.dump将coco_format数据结构写入JSON文件中, 并使用indent=4参数使JSON文件更加易读(每层缩进4个空格).
        json.dump(coco_format, json_file, indent=4)

if __name__ == "__main__":
    # 定义一个包含类别信息的列表, id是类别的唯一标识符, name是类别名称, supercategory是该类别的上一级分类(通常在分类较复杂时使用).
    """
    name: 代表类别的名称, 你可以根据数据集中的实际类别来定义. 例如, 如果你的数据集包含不同类型的表面缺陷, 
    你可以将name设置为具体的缺陷类型, 如"scratch", "crack"等等.
    
    supercategory: 代表类别的上一级分类, 通常用于分类层次较复杂的数据集.
    如果你的数据集只有一层分类, 可以将supercategory设置为与name相同, 或者也可以留空.
    如果你有多个类别, 但都属于同一种大类, 比如"defect"(缺陷), 那么可以将supercategory设置为相同的值, 如"defect".
    """
    categories = [
        {
            "id": 1,
            "name": "defect",
            "supercategory": "defect"
        }
    ]
    # 定义图像文件夹路径和输出JSON文件路径, 并调用convert_to_coco函数
    image_folder = "/Users/ryanlil86/Desktop/database/job/intern/字节跳动/computer_vision/NEU surface defect database"
    output_json = "output_coco_format.json"
    convert_to_coco(image_folder, output_json, categories)
    print("Conversion completed! COCO format JSON saved at:", output_json)
