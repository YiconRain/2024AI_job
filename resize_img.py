#将512*512的jpg转化为224*224的
import os
from PIL import Image

# 设置目标尺寸
target_size = (224, 224)

# 遍历当前目录下的所有文件
for filename in os.listdir('.'):
    # 检查文件扩展名是否为.jpg
    if filename.lower().endswith('.jpg'):
        # 构建完整的文件路径
        file_path = os.path.join('.', filename)
        
        # 打开图片
        with Image.open(file_path) as img:
            # 转换图片大小
            img_resized = img.resize(target_size)
            
            # 保存转换后的图片，可以选择覆盖原图或者保存为新文件
            # img_resized.save(filename)  # 覆盖原图
            img_resized.save(f'{filename}')  # 保存为新文件

print("所有.jpg格式的图片已转换为224*224大小。")