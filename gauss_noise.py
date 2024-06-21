import os
import numpy as np
from PIL import Image

def add_gaussian_noise(image_path, mean=0, variance=30):
    # 打开图片
    img = Image.open(image_path)
    img_array = np.array(img)

    # 添加高斯噪声
    noise = np.random.normal(mean, variance, img_array.shape)
    noisy_img_array = img_array + noise

    # 确保像素值在0-255范围内
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)

    # 将噪声图像转换回PIL图像
    noisy_img = Image.fromarray(noisy_img_array)

    return noisy_img

def process_images_in_directory(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)
            noisy_img = add_gaussian_noise(file_path)

            # 保存带有噪声的图片
            save_path = os.path.join(directory, filename)
            noisy_img.save(save_path)
            print(f'Saved noisy image to {save_path}')

# 调用函数处理当前目录下的图片
process_images_in_directory('./violence_224/test3')