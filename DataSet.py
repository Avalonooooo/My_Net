import os
import random
import torch
from torchvision import transforms
import torch.utils.data as data
# import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
import glob

import argparse

class snow_dataset(data.Dataset):

    def __init__(self, gt_root, mask_root, synthetic_root, is_crop=True, mode = 'train'):
        self.gt_root = gt_root
        self.mask_root = mask_root
        self.synthetic_root = synthetic_root
        self.is_crop = is_crop
        self.mode = mode
        self.imgs_list = os.listdir(gt_root)
        # self.imgs_list.sort(key=lambda x:int(x[:-4]))
        # self.imgs_list = self.imgs_li.sort()

        # self.imgs_list = sorted(glob.glob(gt_root + "/*.*"))


        # print(type(self.imgs_list))

    def __getitem__(self, index):
        img_name = self.imgs_list[index]
        # print (img_name)
        gt_path = os.path.join(self.gt_root, img_name)
        mask_path = os.path.join(self.mask_root, img_name)
        synthetic_path = os.path.join(self.synthetic_root, img_name)

        # read images
        gt_data = Image.open(gt_path).convert('RGB')
        mask_data = Image.open(mask_path).convert('RGB')
        # img = mask_data.load()
        # print (img.shape)
        # print ('mask_data   ',img[0])
        synthetic_data = Image.open(synthetic_path).convert('RGB')

        # totensor and random crop
        toTensor = transforms.ToTensor()
        gt_tensor = toTensor(gt_data)
        mask_tensor = toTensor(mask_data)
        synthetic_tensor = toTensor(synthetic_data)

        if self.is_crop:
            h, w = gt_tensor.shape[1:]
            y = random.randint(0, h - 64)
            x = random.randint(0, w - 64)
            gt_tensor = gt_tensor[:, y:y + 64, x:x + 64]
            mask_tensor = mask_tensor[:, y:y + 64, x:x + 64]
            # mask_tensor = mask_tensor.clamp(0,255)
            # print (mask_tensor)
            # mask_tensor = mask_tensor * 255
            hh, ww = mask_tensor.shape[1:]
            mask_heavy = torch.zeros((hh, ww))
            mask_light = torch.zeros((hh, ww))
            for i in range(0,hh):
                for j in range(0,ww):
                    if (mask_tensor[:,i,j][0]>=0.25):
                        mask_heavy[i, j] = 1
                        # mask_light[i, j] = 1
                    # elif (mask_tensor[:,i,j][0]<0.48 and mask_tensor[:,i,j][0]>=0.17):
                    #         mask_light[i,j] = 1
            # mask_tensor = mask_tensor[:, y:y + 64, x:x + 64]
            synthetic_tensor = synthetic_tensor[:, y:y + 64, x:x + 64]

            return  synthetic_tensor, gt_tensor, mask_heavy, mask_tensor

    def __len__(self):
        return len(self.imgs_list)
