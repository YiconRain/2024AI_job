from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import ViolenceClassifier
from dataset import CustomDataModule

gpu_id = [1]
batch_size = 128
log_name = "resnet18_pretrain"

ckpt_version=1
ckpt_epoch=33
ckpt_loss=0.03

test_num = 1

data_module = CustomDataModule(batch_size=batch_size,test_num=test_num)

ckpt_root = "./train_logs"
ckpt_path = ckpt_root + f'/resnet18_pretrain_test/version_{ckpt_version}/checkpoints/resnet18_pretrain_test-epoch={ckpt_epoch}-val_loss={ckpt_loss}.ckpt'
logger = TensorBoardLogger("test_logs", name=log_name)

model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
trainer = Trainer(
    accelerator='gpu', 
    devices=gpu_id,
    logger=logger)
trainer.test(model, data_module) 