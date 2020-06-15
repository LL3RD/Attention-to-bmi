import sys

sys.path.append('/home/benkesheng/BMI_DETECT/')

import torch
import torch.nn as nn

from torchvision import models
import numpy as np

import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)
print(random.random())
# x = models.densenet121()
# from TypeNet import *
# print(SKDensenet121())


# summary(model,  (3, 224, 224))
# torch.rand((1,3,224,224))

# Pred_Net = models.AlexNet()
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam([
#     {'params': Pred_Net.parameters()}
# ], lr=0.0001)
#
# print(optimizer.state_dict()['param_groups'][0])


# END_EPOCH = 0
#
# IMG_MEAN = [0.485, 0.456, 0.406]
# IMG_STD = [0.229, 0.224, 0.225]
# DEVICE = torch.device("cuda:1")
# IMG_SIZE = 224
# BATCH_SIZE = 64
#
#
# transform = transforms.Compose([
#     Resize(IMG_SIZE),
#     transforms.Pad(IMG_SIZE),
#     transforms.CenterCrop(IMG_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize(IMG_MEAN, IMG_STD)
# ])
#
# dataset = OurDatasets('/home/benkesheng/BMI_DETECT/datasets/Image_ex', mode='3C')
# # val_dataset = Dataset('/home/benkesheng/BMI_DETECT/datasets/Image_val', transform)
# test_dataset = OurDatasets('/home/benkesheng/BMI_DETECT/datasets/Image_test', mode='3C')
#
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
#
# Pred_Net = models.resnet101(pretrained=True)
# for param in Pred_Net.parameters():
#     param.requires_grad = True
# Pred_Net.fc = nn.Sequential(
#     nn.Linear(2048, 1024),
#     nn.ReLU(True),
#     nn.Linear(1024, 512),
#     nn.ReLU(True),
#     nn.Linear(512, 256),
#     nn.ReLU(True),
#     nn.Linear(256, 20),
#     nn.ReLU(True),
#     nn.Linear(20, 1)
# )
#
# Pred_Net = Pred_Net.to(DEVICE)
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam([
#     {'params': Pred_Net.parameters()}
# ], lr=0.0001)
#
#
# def test(model, device, test_loader):
#     model.eval()
#     pred = []
#     targ = []
#     with torch.no_grad():
#         for i, (x,y) in enumerate(test_loader):
#             x, y = x.to(device), y.to(device)
#             # optimizer.zero_grad()
#             y_pred = model(x)
#             pred.append(y_pred.item())
#             targ.append(y.item())
#     MAE = mean_absolute_error(targ, pred)
#     MAPE = mean_absolute_percentage_error(targ, pred)
#     print('\nTest MAE:{}\t Test MAPE:{}'.format(MAE, MAPE))
#     return MAE, MAPE
#
#
# trainer = Trainer(Pred_Net, DEVICE, optimizer, criterion, save_dir=None)
# trainer.test(val_loader)
# MIN_MAE, MAPE = test(Pred_Net, DEVICE, val_loader)
