from utils import *
from TypeNet import *
import torch.optim as optim
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'


save_dir = '/home/ungraduate/hjj/BMI_DETECT/NewExperiment/Models/'
dataset_root = '/home/ungraduate/hjj/BMI_DETECT/datasets'
Nets_save_dir = ['DenseRand_']
DEVICE = torch.device("cuda:3")
train_dataset = OurDatasets(os.path.join(dataset_root, 'Image_train'), mode='3CWithMask', set='Our')
test_dataset = OurDatasets(os.path.join(dataset_root, 'Image_test'), mode='3CWithMask', set='Our')
val_dataset = OurDatasets(os.path.join(dataset_root, 'Image_val'), mode='3CWithMask', set='Our')
for i in range(10):
    setup_seed(i)
    Nets = [Densenet121()]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    criterion = nn.MSELoss().to(DEVICE)
    # criterion = nn.MSELoss().cuda()

    for model, Dir in zip(Nets,Nets_save_dir):
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model,device_ids=[2,1,3])
            # model = model.to(DEVICE)
        # Pred_dict = models.densenet121(pretrained=True).state_dict()
        # model_dict = model.state_dict()
        # Pred_dict = {k: v for k, v in Pred_dict.items() if k in model_dict}
        # model_dict.update(Pred_dict)
        # model.load_state_dict(model_dict)
        Dir = Dir + str(i)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001,
                               weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
        trainer = Trainer(model, DEVICE, optimizer, criterion, save_dir=save_dir + Dir, mult_gpu=False)
        trainer.Loop(50,train_loader, val_loader, scheduler)
        trainer.test(test_loader, sex='diff')
