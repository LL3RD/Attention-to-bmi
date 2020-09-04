# 002270_M_34_198120_10387266


# Test SELayer
import torch
import pretrainedmodels
import pretrainedmodels.utils as utils
model_name = 'se_resnet101' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()
print(model)
load_img = utils.LoadImage()

tf_img = utils.TransformImage(model)

path_img = '/home/benkesheng/BMI_DETECT/datasets/Image_train/002270_M_34_198120_10387266.jpg'
input_img = load_img(path_img)
input_tensor = tf_img(input_img)
input_tensor = input_tensor.unsqueeze(0)


class LayerActivations:
    features = None

    def __init__(self, model):
        self.hook = model.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()

se_out = LayerActivations(model.layer1[1].se_module.sigmoid)
output_logits = model(input_tensor)
se_out.remove()
xs = torch.squeeze(se_out.features.detach()).numpy()
print(xs)
# print(output_logits)
# print(output_logits.shape)









'''
from utils import *
from TypeNet import *
import matplotlib.pyplot as plt

IMG_SIZE = 225
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
            transforms.ToPILImage(),
            Resize(IMG_SIZE),
            transforms.Pad(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])


class LayerActivations:
    features = None

    def __init__(self, model):
        self.hook = model.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


img = cv2.imread('/home/benkesheng/BMI_DETECT/datasets/Image_train/002270_M_34_198120_10387266.jpg')[:, :, ::-1]
img_mask = cv2.imread('/home/benkesheng/BMI_DETECT/datasets/Image_train_mask/Mask_002270_M_34_198120_10387266.jpg')
img_c = img * (img_mask // 255 == 0)
img_c = transform(img_c)


model = SEDensenet121()
# model = SEResnext101()
# print(model)
# pred_dict = torch.load('/home/benkesheng/BMI_DETECT/NewExperiment/Models/Densenet121_3CWithMask/model_epoch_50.ckpt')['state_dict']
pred_dict = torch.load('/home/ungraduate/hjj/BMI_DETECT/NewExperiment/Models/SEDensenet121_3CWithMask_128_tran/model_epoch_50.ckpt')['state_dict']
model.load_state_dict(pred_dict)
print(model)
print(model.features.denseblock2.denselayer1)
se_out = LayerActivations(model.features.SELayer1.fc)# (model.features.denseblock4.denselayer1.relu1)  #
out = model(torch.unsqueeze(img_c, dim=0))
se_out.remove()
xs = torch.squeeze(se_out.features.detach()).numpy()
print(xs.shape)
print(max(xs)-min(xs))
attention = np.argsort(xs)
conv_out = LayerActivations(model.features.denseblock4.denselayer1.relu1)
out = model(torch.unsqueeze(img_c, dim=0))
conv_out.remove()
xs = torch.squeeze(conv_out.features.detach()).numpy()
print(xs.shape)

fig, ax = plt.subplots(figsize=(10, 10))
plt.axis('off')
fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0, left=None, bottom=None, right=None, top=None)
for i,x in enumerate(xs):
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(xs[attention[i+448]])
    if i == 63:
        break
fig.savefig('selayer2mapvisual.jpg')
#
plt.show()



'''