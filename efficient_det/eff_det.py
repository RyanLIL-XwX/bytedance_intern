import tensorflow as tf
from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.modeling.backbones import efficientnet
from official.vision.beta.modeling import factory

# 定义数据集解析器
def parse_coco_data(image_size):
    def parse_fn(value):
        decoder = tf_example_decoder.TfExampleDecoder(
            include_mask=False
        )
        example = decoder.decode(value)
        image = preprocess_ops.normalize_image(example['image'])
        image = preprocess_ops.resize_and_crop_image(
            image,
            image_size,
            image_size,
            aug_scale_min=1.0,
            aug_scale_max=1.0
        )
        return image, example['groundtruth_boxes'], example['groundtruth_classes']
    return parse_fn

# EfficientDet 相关配置
model_config = {
    'input_size': 640,  # 输入图像大小
    'num_classes': 1,   # 类别数量，根据您的数据集调整
    'backbone': 'efficientnetv2-b0',  # 使用 EfficientNet 作为特征提取
    'aspect_ratios': [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]  # 锚框的长宽比
}

# 创建 EfficientDet 模型
def create_efficientdet_model(config):
    backbone = efficientnet.EfficientNet.from_name(config['backbone'])
    model = factory.efficientdet_model(
        backbone=backbone,
        input_size=config['input_size'],
        num_classes=config['num_classes'],
        aspect_ratios=config['aspect_ratios']
    )
    return model

# COCO 数据集的加载路径
train_json = '/Users/ryanlil86/Desktop/database/job/intern/字节跳动/EfficientDet/train_annotations.json'
val_json = '/Users/ryanlil86/Desktop/database/job/intern/字节跳动/EfficientDet/val_annotations.json'
train_image_dir = '/Users/ryanlil86/Desktop/database/job/intern/字节跳动/EfficientDet/NEU surface defect database/train/images'
val_image_dir = '/Users/ryanlil86/Desktop/database/job/intern/字节跳动/EfficientDet/NEU surface defect database/val/images'

# 定义训练集和验证集
train_dataset = tf.data.TFRecordDataset(train_json)
train_dataset = train_dataset.map(parse_coco_data(model_config['input_size'], train_json, train_image_dir))
val_dataset = tf.data.TFRecordDataset(val_json)
val_dataset = val_dataset.map(parse_coco_data(model_config['input_size'], val_json, val_image_dir))

# 创建 EfficientDet 模型
model = create_efficientdet_model(model_config)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=model.compute_loss,
    metrics=['accuracy']
)

# 训练模型
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    steps_per_epoch=100,
    validation_steps=50
)

# 保存模型
model.save('/Users/ryanlil86/Desktop/database/job/intern/字节跳动/EfficientDet/efficientdet_model_result')
