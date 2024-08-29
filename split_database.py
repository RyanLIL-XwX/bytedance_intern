import os # 提供与操作系统交互的功能, 如文件和目录的操作
import shutil # 提供高级文件操作功能, 如复制、移动和删除文件及目录
import random # 提供生成随机数的功能

# 数据集路径
dataset_path = "/Users/ryanlil86/Desktop/database/job/intern/字节跳动/YOLOv8/NEU surface defect database"
train_path = os.path.join(dataset_path, "train") # 定义训练集的路径, 新创建的训练集文件夹将位于dataset_path下的train子目录中
val_path = os.path.join(dataset_path, "val") # 定义验证集的路径, 新创建的验证集文件夹将位于dataset_path下的val子目录中

# 获取所有图片文件(默认都是是.jpg和.bmp文件)
image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.bmp'))]

# 随机打乱文件列表, 以确保数据集划分是随机的, 防止训练集和验证集之间存在系统性的差异
random.shuffle(image_files)

# 计算文件列表的划分点, 这里按80%的比例将数据划分为训练集. split_index表示训练集的大小, 剩下的20%文件将作为验证集
split_index = int(len(image_files) * 0.8)

# 分配文件到训练集和验证集
train_files = image_files[:split_index] # 包含80%的图像文件, 用于训练集
val_files = image_files[split_index:] # 包含20%的图像文件, 用于验证集

# 移动文件到对应的文件夹
for file_name in train_files:
    # shutil.move()函数用于移动文件或目录, 将文件从dataset_path移动到train_path下的images子目录中
    shutil.move(os.path.join(dataset_path, file_name), os.path.join(train_path, "images", file_name))
    # 移动对应的标签文件, 将标签文件从dataset_path移动到train_path下的labels子目录中
    label_file = file_name.replace('.jpg', '.txt').replace('.bmp', '.txt')
    if (os.path.exists(os.path.join(dataset_path, label_file))):
        shutil.move(os.path.join(dataset_path, label_file), os.path.join(train_path, "labels", label_file))

# 移动验证集文件
for file_name in val_files:
    # shutil.move()函数用于移动文件或目录, 将文件从dataset_path移动到val_path下的images子目录中
    shutil.move(os.path.join(dataset_path, file_name), os.path.join(val_path, "images", file_name))
    # 移动对应的标签文件, 将标签文件从dataset_path移动到val_path下的labels子目录中
    label_file = file_name.replace('.jpg', '.txt').replace('.bmp', '.txt')
    if (os.path.exists(os.path.join(dataset_path, label_file))):
        shutil.move(os.path.join(dataset_path, label_file), os.path.join(val_path, "labels", label_file))
