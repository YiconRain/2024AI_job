# 2024AI_job

### classify接口调用方法

```
python classify.py --classify img_path
```

具体示例如：

我将对在/home/rain/wait_classify/文件夹中的图片进行分类，则有

```
img_path=/home/rain/wait_classify
调用：python classify.py --classify /home/rain/wait_classify
```

具体示例如：

<img src="F:\大作业集合\ai导论\img\1718971294267.png" alt="1718971294267" style="zoom:50%;" />

补充说明说明：violence_224文件夹和classify.py文件处于同一目录。

调用：

```
python classify.py --classify ./violence_224/small_test
```

结果：

```
[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
```

### test.py说明

​	violence_224文件夹中有3个测试集，test1、test2、test3，分别对应同源测试集、AIGC测试集、加噪测试集。

​	不同测试集的切换方法是：**修改test.py文件中的test_num**。

### 代码文件说明：

1. dataset.py：提供DataSet、DataLoader等数据工具供模型使用
2. model.py：加载预训练的resnet18模型，定义本次人物中使用loss_fn为交叉熵损失函数，确定前向过程、优化器、trainning_step、validaion_step、test_step等训练工具。
3. train.py：加载Logger和Trainer，使用trainer.train方法训练，保存一个最佳的checkpoint。
4. test.py：加载Trainer和logger，使用trainer.test方法进行测试。
5. gauss_noise.py：给图片添加高斯噪声，用于制作test3
6. rename.py：将stable diffusion生成的图片重命名（打上标签）
7. resize_img.py：将[512,512]的图片转换为[224,224]
