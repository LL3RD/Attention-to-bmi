import sys
sys.path.append('/home/benkesheng/BMI_DETECT/')
from Visual import Visual
from Detected import Image_Processor
import cv2
import matplotlib.pyplot as plt

from utils import *

setup_seed(20)

mask_model = "/home/benkesheng/BMI_DETECT/pose2seg_release.pkl"
keypoints_model = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
# P = Image_Processor(mask_model, keypoints_model)

Detect_Path = '/home/benkesheng/BMI_DETECT/datasets/'
datasets = ['Image_train', 'Image_val', 'Image_test']
mask_datasets = ['Image_train_mask', 'Image_val_mask', 'Image_test_mask']

for set in datasets:
    img_names = os.listdir(Detect_Path+set)
    for img_name in img_names:
        mask_name = 'Mask_'+img_name
        img_path = os.path.join(Detect_Path,set,img_name)
        img_mask_path = os.path.join(Detect_Path,set+'_mask',mask_name)
        img = cv2.imread(img_path)[:,:,::-1]
        img_mask = cv2.imread(img_mask_path)
        img_mask = img_mask//255
        print(img_mask.shape)
        img_c = img*(img_mask==0)
        plt.imshow(img_c)
        plt.show()
        break
    break


'''
Create the Mask channel

for set, maskset in zip(datasets, mask_datasets):
    set_path = os.path.join(Detect_Path, set)
    maskset_path = os.path.join(Detect_Path, maskset)
    img_names = os.listdir(set_path)

    for img_name in img_names:
        img_path = os.path.join(set_path, img_name)
        img = cv2.imread(img_path)
        img_name_mask = 'Mask_'+img_name
        img_path_mask = os.path.join(maskset_path, img_name_mask)

        k, m = P._detected(img)
        shape2d = (m.shape[0], m.shape[1])
        img_f = np.zeros(shape2d + (3,), dtype="float32")
        img_f[:, :, :3] = [255, 255, 255]
        v = Visual(img_f)
        v.draw_keypoints_predictions(k)
        v.draw_masks_predictions(m)

        image = v.image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        cv2.imwrite(img_path_mask, image)
'''



