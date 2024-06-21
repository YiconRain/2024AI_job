from email.mime import image
from numpy import imag
import torch
import torchvision
from trimesh import transform_points
from model import ViolenceClassifier
from PIL import Image
import os
import argparse
import sys
from torchvision import transforms

class ViolenceClass:
    def __init__(self, gpu_enable=True):
        # 加载模型
        ckpt_version=1
        ckpt_epoch=33
        ckpt_loss=0.03
        ckpt_root = "./train_logs"
        ckpt_path = ckpt_root + f'/resnet18_pretrain_test/version_{ckpt_version}/checkpoints/resnet18_pretrain_test-epoch={ckpt_epoch}-val_loss={ckpt_loss}.ckpt'
        self.model = ViolenceClassifier.load_from_checkpoint(checkpoint_path=ckpt_path)
        self.model.freeze() #控制训练好的参数不再变动
        
        #首先得确定模型运行的cpu/gpu
        self.device = torch.device("cuda:1" if gpu_enable and torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    
    def classify(self, imgs : torch.Tensor) -> list:
        # 图像分类
        #走一个forward过程，得到一个[logits,lable]作为元素形成的tensor
        imgs = imgs.to(self.device)
        vio_possibility = self.model(imgs) 
        print(vio_possibility.shape)
        _, preds =torch.max(vio_possibility,1)#找到对应的预测类别
        print(preds.shape)
        return preds.cpu().tolist()#结果运转回到cpu
        #return preds.tolist()
    
    #transAndclassify，接收一个图片路径，将路径下的所有图片，转化为tensor的形式传递给classify函数
    def transAndclassify(self,imgs_dir):
        #定义一个transform操作，把图片转换为tensor并归一化
        transform = transforms.Compose([
            transforms.ToTensor(),
            #归一化，为了简单这里就用基于ImageNet得到的标准化参数
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensors = [] #用来存储单个图片张量
        #遍历目录下的所有文件
        for imgname in os.listdir(imgs_dir):
            img_path = os.path.join(imgs_dir,imgname) #获取图片完整路径
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image) #单个图片tensor
            image_tensors.append(image_tensor)
        #把图片张量列表堆叠成一个批次张量
        batch_tensor = torch.stack(image_tensors).to(self.device)
        return self.classify(batch_tensor)
        

    
def main(args):
    gpu_enable = True
    #初始化一个模型
    the_violence_model = ViolenceClass(gpu_enable=gpu_enable)
    
    #提供测试集上的
    if args.classify:
        preds=the_violence_model.transAndclassify(args.classify)
        print(preds)
        
if __name__ == "__main__":
    #解析命令行的参数,这里仅提供classify功能
    parser = argparse.ArgumentParser(description='Violence Detection Script')
    #添加classify命令
    #但需要用户输入图像所在文件夹的路径，如1.jpg的绝对路径是/home/rain/1.jpg,就需要提供/home/rain
    parser.add_argument('--classify', type=str, help='图像分类操作', metavar='IMAGE_PATH')
    #解析命令行参数
    args = parser.parse_args()
    main(args)
