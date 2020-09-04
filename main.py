from TypeNet import *
import torch.nn as nn
import torch.optim as optim
import torch
from utils import *
import os
import argparse
from torchvision import models
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch BMI')

parser.add_argument('--datasetmode', default='3CWithMask', type=str, help='Type of dataset')
parser.add_argument('--set', default='Our', type=str,
                    help='Dataset to use.')
parser.add_argument('--root', default='/home/ungraduate/hjj/BMI_DETECT/datasets', type=str,
                    help='Path to Dataset.')
parser.add_argument('--save-dir', default='/home/ungraduate/hjj/BMI_DETECT/NewExperiment/Models/9-1NewEX/Densenet121_3CWithMask_2048-1_predtrained', type=str,
                    help='path to save models and state')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU id to use.')
parser.add_argument('--kpts', default=False, type=bool,
                    help='keypoints.')


def main():
    args = parser.parse_args()
    setup_seed(args.seed)
    train_loader, val_loader, test_loader = get_dataloader(args.batch_size, args)
    ''' Visual
    for data, target in val_loader:
        print(data.shape)
        plt.imshow(torch.squeeze(data).permute(1,2,0))
        plt.show()
        break
    return
    '''
    # print(len(train_loader))
    Pred_dict = torch.load('/home/benkesheng/BMI_DETECT/ReDone_CSV/model/densenet121.pkl')
    # Pred_dict = torch.load('/home/benkesheng/BMI_DETECT/ReDone_CSV/model/Ours.pkl')
    # Pred_dict = torch.load('/home/ungraduate/hjj/BMI_DETECT/NewExperiment/Models/SEDensenet121_3CWithMask_128_tran/model_epoch_50.ckpt')['state_dict']
    # Pred_dict = models.densenet121(pretrained=True).state_dict()
    model = Densenet121()
    # print(model)
    # model.load_state_dict(Pred_dict)

    model_dict = model.state_dict()
    Pred_dict = {k: v for k, v in Pred_dict.items() if k in model_dict and (
                k != 'classifier.0.weight' and k != 'classifier.0.bias')}# and k != 'classifier.6.weight')}  # and k != 'features.conv0.weight')}
    model_dict.update(Pred_dict)
    model.load_state_dict(model_dict)

    DEVICE = torch.device("cuda:" + args.gpu)

    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)


    trainer = Trainer(model, DEVICE, optimizer, criterion, save_dir=args.save_dir)
    # trainer.load('/home/benkesheng/BMI_DETECT/NewExperiment/Models/SEDensenet121_3CWithMask_128_tran/model_epoch_50.ckpt')
    trainer.Loop(args.epochs, train_loader, val_loader, scheduler)
    trainer.test(test_loader, sex='diff')


if __name__ == '__main__':
    main()
