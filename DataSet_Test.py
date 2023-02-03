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

    def __init__(self, gt_root, synthetic_root, is_crop=True, mode = 'train'):
        self.gt_root = gt_root
        self.synthetic_root = synthetic_root
        self.is_crop = is_crop
        self.mode = mode
        self.imgs_list = os.listdir(gt_root)
        self.imgs_list.sort(key=lambda x:int(x[:-4]))
        # self.imgs_list = self.imgs_li.sort()

        # self.imgs_list = sorted(glob.glob(gt_root + "/*.*"))


        # print(type(self.imgs_list))

    def __getitem__(self, index):
        img_name = self.imgs_list[index]
        # print (img_name)
        gt_path = os.path.join(self.gt_root, img_name)
        synthetic_path = os.path.join(self.synthetic_root, img_name)

        # read images
        gt_data = Image.open(gt_path).convert('RGB')
        # img = mask_data.load()
        # print (img.shape)
        # print ('mask_data   ',img[0])
        synthetic_data = Image.open(synthetic_path).convert('RGB')

        # totensor and random crop
        toTensor = transforms.ToTensor()
        # resize = transforms.RandomCrop((256,256))
        gt_tensor = toTensor(gt_data)
        # gt_tensor = resize(gt_tensor)
        synthetic_tensor = toTensor(synthetic_data)
        # synthetic_tensor = resize(synthetic_tensor)


        if self.mode == 'train':
            return  synthetic_tensor, gt_tensor
        # mask_tensor,

    def __len__(self):
        return len(self.imgs_list)
