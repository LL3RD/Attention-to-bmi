from OurDatasets import *
from TypeNet import *
from utils.utils import mean_absolute_error, mean_absolute_percentage_error
from train import AverageMeter

model = SEDensenet121()
model.load_state_dict(torch.load(
    '/home/ungraduate/hjj/BMI_DETECT/NewExperiment/Models/SEDensenet121_3CWithMask_128_tran/model_epoch_50.ckpt')[
                          'state_dict'])
dataset = OurDatasets('/home/ungraduate/hjj/BMI_DETECT/datasets/Image_test', mode='3CWithMask', set='Our')
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
model.eval()

under_mae = AverageMeter('Under_MAE')
under_mape = AverageMeter('Under_MAPE')
normal_mae = AverageMeter('Normal_MAE')
normal_mape = AverageMeter('Normal_MAPE')
over_mae = AverageMeter('Over_MAE')
over_mape = AverageMeter('Over_MAPE')
obese_mae = AverageMeter('Obese_MAE')
obese_mape = AverageMeter('Obese_MAPE')

with torch.no_grad(): 
    for img, (sex, targ) in test_loader:
        out = model(img)
        out = out.detach().cpu().numpy()
        target = targ.detach().cpu().numpy()
        mae = mean_absolute_error(target, out)
        mape = mean_absolute_percentage_error(target,out)
        if target <= 18.5:
            under_mae.update(mae)
            under_mape.update(mape)
        elif target > 18.5 and target <= 25:
            normal_mae.update(mae)
            normal_mape.update(mape)
        elif target > 25 and target <= 30:
            over_mae.update(mae)
            over_mape.update(mape)
        elif target > 30:
            obese_mae.update(mae)
            obese_mape.update(mape)

print("UnderMAE:", under_mae.avg, '\tUnderMAPE:', under_mape.avg)
print("NormalMAE:", normal_mae.avg, '\tNormalMAPE:', normal_mape.avg)
print("OverMAE:", over_mae.avg, '\tOverMAPE:', over_mape.avg)
print("ObeseMAE:", obese_mae.avg, '\tObeseMAPE:', obese_mape.avg)
