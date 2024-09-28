import pandas as pd
import torch, random, time, os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob, tqdm
from PIL import Image

from preprocess import preprocess
from net import FlowersMode
from utils.load import load_model
from utils.fit import fit_one_epoch
from loss import lossFunc



# ---------------------------------------------------------------
# Hyper Parameters
BATCH_SIZE = 4
# 初始学习率大小
warmup = False
warmup_lr = 1e-6
basic_lr_per_img = 0.01 / 64.0
LR = basic_lr_per_img * BATCH_SIZE
use_cosine = False
# 训练的世代数
warmup_epoch = 1
start_epoch = 0
EPOCH = 30
# 图片原来的size
pic_shape = 256
# 网络输入图片size
RE_SIZE_shape = 192
# 总的类别数
num_classes = 14
# 标签平滑
label_smoothing = 0
CUDA = True
# 是否载入预训练模型参数
use_pretrain = False
#有多个gpu才能为True
Parallel = False
# gpu
gpu_device_id = 0
# ---------------------------------------------------------------

device = torch.device("cuda:%d" % gpu_device_id if torch.cuda.is_available() else "cpu")

# 数据集读取
train_loader = DataLoader(
    dataset=preprocess.flowerDataset(x_train_path, y_train, RE_SIZE_shape, num_classes, train=True),
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=1,
    pin_memory=False,
    drop_last=True
)
val_loader = DataLoader(
    dataset=preprocess.flowerDataset(x_test_path, y_test, RE_SIZE_shape, num_classes, train=False),
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=1,
    pin_memory=False,
    drop_last=True
)

for x, y in train_loader:
    print(y)


model = FlowersMode.modeFlowers(num_classes=num_classes)

model_path = ''
#torch.nn.init.normal(model.weights.data, mean=0, std=1)
if use_pretrain:
    LR = 7.61e-6
    model_path = '/home/b201/gzx/yolox_self/logs/' \
                 'val_loss3.507-size640-lr0.00000761-ep055-train_loss3.465.pth'
    load_model(model, model_path)

# 冻结主干进行训练
# for param in model.backbone.parameters():
#     param.requires_grad = False
if Parallel:
    model = torch.nn.DataParallel(model)

if CUDA:
    model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
# loss_func = loss_baseBox.YOLOLoss(num_classes=num_classes)
if not use_cosine:
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
else:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5, eta_min=LR/100)

loss_func = lossFunc.flowersLoss(input_shape=RE_SIZE_shape, num_classes=num_classes, label_smoothing=label_smoothing, device=device)

val_loss_save = 1e10
# time = time.asctime(time.localtime(time.time()))
# logs_save_path = '/home/b201/gzx/yolox_self/logs/' + str(time)
# os.mkdir(logs_save_path)
# 预热训练
if not use_pretrain and warmup:
    print('start warm up Training!')
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    optimizer_wp = torch.optim.Adam(model.parameters(), lr=warmup_lr, weight_decay=5e-4)
    lr_scheduler_wp = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_wp, gamma=(LR/warmup_lr)**(1/warmup_epoch))
    for epoch in range(warmup_epoch):
        val_loss_save = fit_one_epoch(model, optimizer_wp, loss_func, lr_scheduler_wp, warmup_epoch, epoch, train_loader, val_loader, RE_SIZE_shape, val_loss_save, time, CUDA, device, warmup=True)
    print('Finish warm up Training!')
# 正式训练
val_loss_save = 1e10
print('start Training!')
for epoch in range(start_epoch, EPOCH):
    val_loss_save = fit_one_epoch(model, optimizer, loss_func, lr_scheduler, EPOCH, epoch, train_loader, val_loader, RE_SIZE_shape, val_loss_save, time, CUDA, device)

