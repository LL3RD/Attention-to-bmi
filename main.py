from TypeNet import CBAMResnet50, CBAMResnet101, CBAMDensenet121, SEResnet101,SEDensenet121, SKNet101, Resnet101
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
import os
import argparse
from torchvision import models

parser = argparse.ArgumentParser(description='PyTorch BMI')
setup_seed(0)

parser.add_argument('--datasetmode',default='4C',type=str,help='Type of dataset')
parser.add_argument('--save-dir',default='Rensenet101_4C',type=str,help='path to save models and state')


parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
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


def get_dataloader(batch_size, args, root='/home/benkesheng/BMI_DETECT/datasets'):
    train_dataset = OurDatasets(os.path.join(root, 'Image_train'), mode=args.datasetmode)
    test_dataset = OurDatasets(os.path.join(root, 'Image_test'), mode=args.datasetmode)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=args.workers)

    return train_loader, val_loader, test_loader


def main():
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloader(args.batch_size,args)

    # Pred_dict = torch.load('/home/benkesheng/BMI_DETECT/ReDone_CSV/model/densenet121.pkl')
    Pred_dict = torch.load('/home/benkesheng/BMI_DETECT/ReDone_CSV/model/Ours.pkl')
    # Pred_dict = models.densenet121(pretrained=True).state_dict()
    model = Resnet101()
    # model.load_state_dict(Pred_dict)
    # print(model)

    model_dict = model.state_dict()
    Pred_dict = {k: v for k, v in Pred_dict.items() if k in model_dict  and (k != 'fc.6.weight' and k != 'fc.6.bias'and k != 'fc.8.weight' and k != 'conv1.weight')}
    model_dict.update(Pred_dict)
    model.load_state_dict(model_dict)

    DEVICE = torch.device("cuda:" + args.gpu)

    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.1)

    trainer = Trainer(model, DEVICE, optimizer, criterion, save_dir=args.save_dir)
    # trainer.load('Densenet121_3CWithMask/model_epoch_40.ckpt')
    trainer.Loop(100, train_loader, val_loader, scheduler)
    trainer.test(test_loader)


if __name__ == '__main__':
    main()
