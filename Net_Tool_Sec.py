import torch
import torch.nn as nn
import torch.nn.functional as F


from Unet_Tool import *

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8,out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8,out_channels),
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8,out_channels),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))




class CSR_Desnow(nn.Module):
    def __init__(self):
        super(CSR_Desnow, self).__init__()

        self.UN = UNet(n_channels=3, bilinear=True)
        self.UN_fir = UNet(n_channels=3, bilinear=True)
        self.UN_sec = UNet(n_channels=3, bilinear=True)
        self.UN_thr = UNet_Four(n_channels=3, bilinear=True)

    def forward(self,x, con_x, Max_fir, Max_sec, Re_thr):

        Fea_thr, A_thr = self.UN_thr(Re_thr, 3)
        Fea_sec, A_sec = self.UN_sec(Max_sec, A_thr, Fea_thr, 2)
        Fea_fir, A_fir, = self.UN_fir(Max_fir, A_sec, Fea_sec, 1)
        Fea, A = self.UN(con_x, A_fir, Fea_fir, 0)



        return  A_fir, A_sec, A_thr, A
class CSR_Net(nn.Module):
    def __init__(self):
        super(CSR_Net, self).__init__()
        self.De_fir = CSR_Desnow()

    def forward(self, x, con_x, snow_fir_img, snow_sec_img, Re_thr):


        A_fir, A_sec, A_thr, A = self.De_fir(x, con_x, snow_fir_img, snow_sec_img, Re_thr)

        return A_fir, A_sec, A_thr, A

