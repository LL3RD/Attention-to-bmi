from TypeNet import *
import torch.nn as nn
import torch.optim as optim
import torch
from utils import *
import os
import argparse
from torchvision import models

parser = argparse.ArgumentParser(description='PyTorch BMI')

parser.add_argument('--datasetmode', default='3CWithMask', type=str, help='Type of dataset')
parser.add_argument('--set', default='Our', type=str,
                    help='Dataset to use.')
parser.add_argument('--root', default='/home/ungraduate/hjj/BMI_DETECT/datasets', type=str,
                    help='Path to Dataset.')
parser.add_argument('--save-dir', default='/home/ungraduate/hjj/BMI_DETECT/NewExperiment/Models/Cache/SEResnext_4DIM', type=str,
                    help='path to save models and state')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default='1', type=str,
                    help='GPU id to use.')
parser.add_argument('--kpts', default=True, type=bool,
                    help='Need kpts features or not')


def main():
    args = parser.parse_args()
    setup_seed(args.seed)
    train_loader, val_loader, test_loader = get_dataloader(args.batch_size, args)
    # for data, targets in train_loader:
    #     F_M, target = targets
    #     if True:
    #         data, kpt = data
    #
    #         print(kpt.shape)
    #         import matplotlib.pyplot as plt
    #         plt.imshow(kpt[2][0])
    #         plt.show()
    #         print(data[2].shape)
    #         data = data[2]
    #         data = data.permute(1,2,0)
    #         plt.imshow(data)
    #         plt.show()
    #         break
    # print(len(train_loader))
    # Pred_dict = torch.load('/home/benkesheng/BMI_DETECT/ReDone_CSV/model/densenet121.pkl')
    # Pred_dict = torch.load('/home/benkesheng/BMI_DETECT/ReDone_CSV/model/Ours.pkl')
    Pred_dict = torch.load('/home/ungraduate/hjj/BMI_DETECT/NewExperiment/Models/Cache/SEResnext/model_epoch_50.ckpt')['state_dict']
    # Pred_dict = models.resnext101_32x8d(pretrained=True).state_dict()
    model = SEResnext101()
    # x = torch.randn(64, 3, 450, 600)
    # y = model(x)
    # print(model)
    # print(KDensenet121())
    # model.load_state_dict(Pred_dict)

    model_dict = model.state_dict()
    Pred_dict = {k: v for k, v in Pred_dict.items() if k in model_dict and (k != 'conv1.weight')}# and k!= 'layer1.0.downsample.0.weight')}

                # k != 'classifier.4.weight' and k != 'classifier.4.bias' and k != 'classifier.6.weight')}  # and k != 'features.conv0.weight')}
    model_dict.update(Pred_dict)
    model.load_state_dict(model_dict)

    DEVICE = torch.device("cuda:" + args.gpu)

    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

    trainer = Trainer(model, DEVICE, optimizer, criterion, save_dir=args.save_dir, kpt_f=args.kpts)
    # trainer.load('Models/Cache/KDensenet121/best_model.ckpt')
    trainer.Loop(args.epochs, train_loader, val_loader, scheduler)
    trainer.test(test_loader, sex='diff')


if __name__ == '__main__':
    main()
