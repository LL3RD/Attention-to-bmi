# 002270_M_34_198120_10387266

from utils import *

transform = transforms.Compose([
            transforms.ToPILImage(),
            Resize(IMG_SIZE),
            transforms.Pad(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])




