import torch.utils.data as data
import torch
from utils import *
import os
import re
import cv2

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def get_dataloader(batch_size, args,):
    train_dataset = OurDatasets(os.path.join(args.root, 'Image_train'), mode=args.datasetmode, set=args.set)
    test_dataset = OurDatasets(os.path.join(args.root, 'Image_test'), mode=args.datasetmode, set=args.set)
    if args.set == 'Our':
        train_size = int(0.8 * len(train_dataset))  
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    elif args.set == 'Author':
        val_dataset = OurDatasets(os.path.join(args.root, 'Image_val'), mode=args.datasetmode, set=args.set)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=args.workers)

    return train_loader, val_loader, test_loader


class OurDatasets(data.Dataset):
    def __init__(self, file, mode, set='Our'):
        self.img_names = os.listdir(file)
        self.file = file
        self.mode = mode
        self.set = set
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            Resize(IMG_SIZE),
            transforms.Pad(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_mask_name = 'Mask_' + img_name
        img = cv2.imread(os.path.join(self.file, img_name), flags=1)[:, :, ::-1]

        if self.mode == '4C':
            img = self.transform(img)
            img = transforms.Normalize(IMG_MEAN, IMG_STD)(img)
            img_mask = cv2.imread(os.path.join(self.file + '_mask', img_mask_name), flags=0)
            img_mask = self.transform(img_mask)
            img_c = torch.cat((img, img_mask), dim=0)

        elif self.mode == '3CWithMask':
            img_mask = cv2.imread(os.path.join(self.file + '_mask', img_mask_name))
            img_c = img * (img_mask // 255 == 0)
            img_c = self.transform(img_c)
            img_c = transforms.Normalize(IMG_MEAN, IMG_STD)(img_c)

        elif self.mode == '3C':
            img = self.transform(img)
            img = transforms.Normalize(IMG_MEAN, IMG_STD)(img)
            img_c = img
        else:
            img_c = None

        if self.set == 'Our':
            ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", img_name)
            sex = 0 if (ret.group(1) == 'F' or ret.group(1) == 'f') else 1
            BMI = torch.from_numpy(np.asarray((int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2))
        elif self.set == 'Author':
            ret = re.match(r"[a-zA-Z0-9]+_[a-zA-Z0-9]+__?(\d+)__?(\d+)__?([a-z]+)_*", img_name)
            height = float(ret.group(2)) * 0.0254
            weight = float(ret.group(1)) * 0.4536
            sex = (lambda x: x == 'false')(ret.group(3))
            BMI = weight / (height ** 2)
        else:
            sex, BMI = None, None

        return img_c, (sex, BMI)


'''
class Datasets_ori(data.Dataset):
    def __init__(self, file):
        self.img_names = os.listdir(file)
        self.file = file

        self.transform = transforms.Compose([
            Resize(IMG_SIZE),
            transforms.Pad(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        Pic = Image.open(os.path.join(self.file, img_name))
        Pic = self.transform(Pic)

        ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", img_name)
        BMI = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2
        return Pic, BMI
'''
