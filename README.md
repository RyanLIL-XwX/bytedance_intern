# NEU表面缺陷数据库转换为COCO格式报告

## 任务概述
本次任务的主要目标是下载NEU表面缺陷数据库，提取其内容，并将其中的BMP图像文件转换为COCO格式。

---

## 步骤 1: 下载NEU表面缺陷数据库
首先，需要获取NEU表面缺陷数据库，该数据库包含用于表面缺陷检测的图像数据。该数据库以压缩的`.rar`格式提供，因此需要解压缩以访问其中的图像数据。

---

## 步骤 2: 解压缩`.rar`文件
由于数据库是压缩在`.rar`文件中的，因此需要使用合适的工具进行解压缩。在研究后，我决定使用通过Homebrew安装的`unar`工具。
并且在压缩后我的到了NEU surface defect database的文件夹，里面全部都是`bmp`格式的图像文件。

- **安装`unar`的命令:**
  ```bash
  brew install unar
  ```
- **使用 `unar`对`.rar`文件进行压缩**
  ```bash
  unar NEU surface defect database.rar # 来对.rar文件进行减压
  ```

## 步骤 3: 将BMP图像转换为COCO格式
在成功解压缩数据库后，接下来的任务是将BMP图像转换为COCO格式(COCO格式是一种结构化的数据格式，用于组织图像数据和标注，适用于目标检测和分割任务的模型训练)。

COCO格式包括几个关键组件：
- Images: 包含每个图像的元数据，例如文件名、高度、宽度和ID。
- Annotations: 描述图像中每个对象的边界框、分割、类别等信息。
- Categories: 定义数据集中不同的对象类别，每个类别都有一个ID、名称和上级类别

我写了一个叫做coco.py的文件来对NEU surface defect database中的`bmp`文件进行转换。
```python
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
```
### 代码解析
脚本的关键部分包括使用的库（如os、cv2、json、PIL.Image和pycocotools.coco），转换过程（脚本遍历指定文件夹中的每个BMP图像，
将其转换为JPEG格式，并收集必要的元数据如图像尺寸和文件名），生成带有占位符数据的标注信息，数据以COCO格式组织并写入到一个JSON文件中。
类别定义为一个ID为1的类别，表示缺陷，这是本数据集的主要关注点。最后通过运行coco.py生成了一个名为output_coco_format.json的JSON文件，其中包含COCO格式的数据，可用于进一步的计算机视觉任务，例如模型训练和评估。
  
