from utils import *
from TypeNet import *
import torch.optim as optim

Nets = [AlexNet(),MobileNet(), GoogleNet(), VGG16(), Resnext101(), ]
save_dir = '/home/ungraduate/hjj/BMI_DETECT/NewExperiment/Models/'
dataset_root = '/home/ungraduate/hjj/BMI_DETECT/datasets'
Nets_save_dir = ['AlexNet','MobileNet','GoogleNet','VGG16','Resnext101']
DEVICE = torch.device("cuda:2")

train_dataset = OurDatasets(os.path.join(dataset_root, 'Image_train'), mode='3CWithMask', set='Our')
test_dataset = OurDatasets(os.path.join(dataset_root, 'Image_test'), mode='3CWithMask', set='Our')
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True,
                                           num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

criterion = nn.MSELoss().to(DEVICE)

for model, Dir in zip(Nets,Nets_save_dir):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001,
                           weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    trainer = Trainer(model, DEVICE, optimizer, criterion, save_dir=save_dir + Dir)
    trainer.Loop(50,train_loader, val_loader, scheduler)
    trainer.test(test_loader, sex='diff')