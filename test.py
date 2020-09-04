import sys

sys.path.append('/home/ungraduate/hjj/BMI_DETECT/')

import torch
import torch.nn as nn
from OurDatasets import *
from TypeNet import *
from torchvision import models
import numpy as np
import random




# model = SEResnext101()
# model.load_state_dict(torch.load(
#     '/home/ungraduate/hjj/BMI_DETECT/NewExperiment/Models/SEResnext101/best_model.ckpt')[
#                           'state_dict'])

root = '/home/ungraduate/hjj/BMI_DETECT/datasets/'
failures = ['000871_F_24_160020_5805983.jpg', '001903_F_29_167640_5805983.jpg', '002267_M_34_193040_5760623.jpg']

for fail in failures:
    img = cv2.imread(root + 'Image_test/' + fail, flags=1)[:, :, ::-1]
    img_mask = cv2.imread(root + 'Image_test_mask/'+'Mask_'+fail)
    img_c = img * (img_mask // 255 == 0)
    cv2.imwrite('FailureDemo/'+'Mask_' + fail, img_c[:,:,::-1])


# def test(model, device, test_loader):
#     model.eval()
#     model = model.to(device)
#     pred = []
#     targ = []
#     with torch.no_grad():
#         for i, (x,(sex,y)) in enumerate(test_loader):
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
# # Failure And Precise
# model = SEDensenet121()
# model.load_state_dict(torch.load(
#     '/home/ungraduate/hjj/BMI_DETECT/NewExperiment/Models/SEDensenet121_3CWithMask_128_tran/model_epoch_50.ckpt')[
#                           'state_dict'])
# dataset = OurDatasets('/home/ungraduate/hjj/BMI_DETECT/datasets','Image_test', mode='3CWithMask', set='Our', partition='test')
# test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
# model.eval()
# # test(model, torch.device("cuda:0"), test_loader)
#
#
#
# import time
# sum = 0
# with torch.no_grad():
#     for img, (name, targ) in test_loader:
#
#         t = time.time()
#         out = model(img)
#         out = out.detach().cpu().numpy()
#         target = targ.detach().cpu().numpy()
#         sum += time.time()-t
#         # print(sum)
#
#         if abs(out - target) >= 10:
#             print(name[0], '\tTruth:', target, '\tPred:', out)
#     print(sum/len(test_loader))

# for x,y in test_loader:
#     print(x.shape)
#     x = torch.squeeze(x).permute(1,2,0)
#     plt.imshow(x)
#     plt.show()
#     break


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
