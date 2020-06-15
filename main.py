from TypeNet import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
import os
import argparse
from torchvision import models



parser = argparse.ArgumentParser(description='PyTorch BMI')

parser.add_argument('--datasetmode', default='3CWithMask', type=str, help='Type of dataset')
parser.add_argument('--save-dir', default='Models/SEDensenet121_3CWithMask_Tran', type=str, help='path to save models and state')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
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
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default='1', type=str,
                    help='GPU id to use.')


def get_dataloader(batch_size, args, root='/home/benkesheng/BMI_DETECT/datasets'):
    train_dataset = OurDatasets(os.path.join(root, 'Image_train'), mode=args.datasetmode)
    test_dataset = OurDatasets(os.path.join(root, 'Image_test'), mode=args.datasetmode)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=args.workers)

    return train_loader, val_loader, test_loader


def main():
    args = parser.parse_args()
    setup_seed(args.seed)
    train_loader, val_loader, test_loader = get_dataloader(args.batch_size, args)

    # Pred_dict = torch.load('/home/benkesheng/BMI_DETECT/ReDone_CSV/model/densenet121.pkl')
    # Pred_dict = torch.load('/home/benkesheng/BMI_DETECT/ReDone_CSV/model/Ours.pkl')
    # Pred_dict = torch.load('Densenet121_3CWithMask/model_epoch_40.ckpt')
    # Pred_dict = models.densenet121(pretrained=True).state_dict()
    model = SEDensenet121()
    print(model)
    # for i in model.parameters():
    #     i.requires_grad = False

    # for k, v in model.named_parameters():
    #     print(k)
        # if k[:18] == 'features.CBAMLayer':
        #     v.requires_grad = True

    # model.load_state_dict(Pred_dict)
    # print(model)

    # model_dict = model.state_dict()
    # Pred_dict = {k: v for k, v in Pred_dict.items() if k in model_dict  and (k != 'classifier.4.weight' and k != 'classifier.4.bias' and k != 'classifier.6.weight')}  # and k != 'conv1.weight')}
    # model_dict.update(Pred_dict)
    # model.load_state_dict(model_dict)

    DEVICE = torch.device("cuda:" + args.gpu)

    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                           weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

    trainer = Trainer(model, DEVICE, optimizer, criterion, save_dir=args.save_dir)
    trainer.load('Models/Densenet121_3CWithMask/model_epoch_50.ckpt')
    trainer.Loop(args.epochs, train_loader, val_loader, scheduler)
    trainer.test(test_loader)


if __name__ == '__main__':
    main()
