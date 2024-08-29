import json # 用于处理JSON文件的读取和解析
import os # 用于文件和目录的操作, 例如创建目录或处理文件路径

# 加载COCO格式的JSON文件, 将文件对象赋值给变量f. 使用with语句可以确保在读取完文件后自动关闭文件
with open('output_coco_format.json') as f:
    # 使用json.load方法将JSON文件的内容解析为Python的字典类型, 并将其赋值给变量data. data包含了COCO格式的所有标注数据
    data = json.load(f)

# 创建名为labels的文件夹, 用于保存转换后的YOLO格式的标注文件. exist_ok=True意味着如果文件夹已经存在, 不会引发错误
os.makedirs('labels', exist_ok=True)

# 遍历data字典中的annotations列表. 每个annotation代表一个标注, 即一个对象的边界框信息及其他相关数据
for annotation in data['annotations']:
    image_id = annotation['image_id'] # 获取当前标注对应的图像ID(image_id), 以便在下一步中找到与该标注对应的图像信息
    # 从data字典的images列表中找到与当前标注的image_id匹配的图像信息, 并将其赋值给image_info. next()函数用于从生成器中获取第一个匹配的元素
    image_info = next(item for item in data['images'] if item['id'] == image_id)
    
    # # 打印调试信息
    # print(f"Processing annotation for image_id: {image_id}")
    # print(f"bbox: {annotation['bbox']}")
    
    # 计算中心点坐标和宽高, 归一化
    # 检查当前标注是否有bbox(边界框)信息. 如果有, 继续执行下面的代码
    """
    将COCO格式中表示边界框的坐标和尺寸转换为YOLO格式所需要的归一化数据.
    'x, y, w, h'从"annotation['bbox']"中解包, 这些值分别表示边界框的左上角的x坐标、
    y坐标、宽度和高度. 然后, 代码计算边界框的中心点坐标: 'x_center'是通过将x坐标加上宽度的一半得到边界框的水平中心, 
    再将其除以图像的宽度进行归一化处理；'y_center'类似地计算垂直中心, 并除以图像高度归一化. 接着, 
    边界框的宽度'w'和高度'h'也分别除以图像的宽度和高度进行归一化. 归一化处理后的'x_center', 'y_center',
    'w', 和'h'将以相对于图像尺寸的比例表示, 这正是YOLO格式所要求的形式.  
    """
    if (annotation['bbox']):
        x, y, w, h = annotation['bbox']
        x_center = (x + w / 2) / image_info['width']
        y_center = (y + h / 2) / image_info['height']
        w = w / image_info['width']
        h = h / image_info['height']
        
        # 生成YOLO格式的标注字符串. YOLO格式包含五个值: 类别ID(注意这里类别ID减1, YOLO格式中的类别ID通常从0开始)、归一化的中心点x坐标、归一化的中心点y坐标、归一化的宽度和归一化的高度.  
        yolo_format = f"{annotation['category_id'] - 1} {x_center} {y_center} {w} {h}"
        # 生成YOLO格式标注文件的路径. 文件名与图像文件名相同, 只是扩展名从.jpg替换为.txt. os.path.join用于确保文件路径的正确性
        label_path = os.path.join('labels', f"{image_info['file_name'].replace('.jpg', '.txt')}")
        
        # # 打印生成的文件路径
        # print(f"Saving to: {label_path}")
        
        # 以追加模式('a')打开或创建标注文件. 如果文件已存在, 新的标注将追加到文件末尾
        with open(label_path, 'a') as label_file:
            # 将YOLO格式的标注写入文件, 每个标注占一行, 并在行末添加换行符
            label_file.write(yolo_format + '\n')
    else:
        pass
