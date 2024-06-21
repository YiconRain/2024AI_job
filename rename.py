#该文件将stable_diffusion生成的图片重命名为和训练集类似的形式，用0表示非暴力，1表示暴力
import os

# 设置图片所在的文件夹路径
folder_path = './'  # 请替换为你的图片文件夹路径

# 初始化计数器
counter = 1

# 获取当前文件夹下所有.jpg文件的列表
jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]

# 如果需要保持原有顺序，可以按文件创建时间排序
# jpg_files.sort(key=lambda x: os.path.getctime(os.path.join(folder_path, x)))

# 遍历.jpg文件列表
for filename in jpg_files:
    # 构造新的文件名，确保扩展名不变
    new_filename = f"1_{counter:04d}.jpg"  # 格式化字符串，确保序号是四位数且扩展名为.jpg

    # 构造完整的旧文件路径和新文件路径
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_filename)

    # 重命名文件
    os.rename(old_file, new_file)
    print(f"Renamed '{filename}' to '{new_filename}'")  # 打印重命名信息

    # 更新计数器
    counter += 1

print("所有.jpg图片重命名完成。")