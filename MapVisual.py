# 002270_M_34_198120_10387266

from utils import *
from TypeNet import *
import matplotlib.pyplot as plt
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


model = CBAMDensenet121()
pred_dict = torch.load('CBAMDensenet121_3CWithMask/best_model.ckpt')['state_dict']
model.load_state_dict(pred_dict)
# print(model.features.denseblock1)
conv_out = LayerActivations(model.features.transition1.conv)# .denseblock1.denselayer6.conv2)  #
out = model(torch.unsqueeze(img_c, dim=0))
conv_out.remove()
xs = torch.squeeze(conv_out.features.detach()).numpy()
print(xs.shape)
fig, ax = plt.subplots(figsize=(10, 10))
plt.axis('off')
fig.tight_layout()
fig.subplots_adjust(wspace = 0, hspace = 0,left=None, bottom=None, right=None, top=None)
for i,x in enumerate(xs):
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(xs[i])
    if i==63:
        break
fig.savefig('kkkk.jpg')

plt.show()

