from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torchvision import models

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(int(out_ch/16),out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(int(out_ch/16),out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(int(out_ch/16),out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(int(F_int/16),F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(int(F_int/16),F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1,1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch=3, output_ch=2):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)
        self.log = nn.LogSoftmax(dim=1)
        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.log(self.Conv(d2))

      #  out = self.active(out)

        return out



def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out


class FeaBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(FeaBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8,outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8,outchannel),
        )

        self.left1 =nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(8,outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(8,outchannel),
        )

        self.left2 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(8,outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(8,outchannel),
        )
        self.shortcut = nn.Sequential()
        # self.p1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        # self.p2 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=3, dilation=3)
        # self.p3 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=5, dilation=5)
        # self.shortcut = nn.Sequential()
        if inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1),
                nn.BatchNorm2d(outchannel),
            )
        self.ReLU= nn.ReLU()
        self.De = nn.Conv2d(3*outchannel, outchannel, kernel_size=1, stride=1)
        self.cbam = CBAM(3*outchannel, 16)
    def forward(self, x):

        # p1 = self.p1(x)
        # p2 = self.p2(x)
        # p3 = self.p3(x)
        x1 = self.left(x)
        x2 = self.left1(x)
        x3 = self.left2(x)
        Cat = torch.cat([x1, x2, x3])
        f = self.De(self.cbam(Cat))

        out = self.ReLU(self.shortcut(x) + f)
        # else:
        #     out = self.ReLU(self.shortcut(x) + x1)

        return out
class FeaNet(nn.Module):
    def __init__(self, inchannel):
        super(FeaNet, self).__init__()
        self.block1 = FeaBlock(64,64)
        self.block2 = FeaBlock(64, 64)
        self.block3 = FeaBlock(64, 64)
        self.block4 = FeaBlock(64, 64)
        self.block5 = FeaBlock(64, 64)
        self.block6 = FeaBlock(64, 64)
        self.Encode = nn.Conv2d(inchannel,64,kernel_size=3,padding=1,stride=1)
    def forward(self, x):
        inc = self.Encode(x)
        f = self.block1(inc)
        f1 = self.block2(f)
        f2 = self.block3(f1)
        f3 = self.block4(f2)
        f4 = self.block5(f3)
        f5 = self.block5(f4)
        return f5

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(int(mid_channels/16),mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(int(out_channels/16),out_channels),
            # nn.ReLU(inplace=True)
        )
        self.double_conv_f = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(int(mid_channels / 16), mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(int(out_channels / 16), out_channels),
            # nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 1),
            nn.GroupNorm(int(out_channels/16),out_channels),
        )
        self.ReLU = nn.ReLU()
    def forward(self, x):
        return self.ReLU(self.double_conv(x) + self.shortcut(x))

# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""
#
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(int(mid_channels/16),mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(int(out_channels/16),out_channels),
#             nn.ReLU(inplace=True)
#         )
#         # self.double_conv_f = nn.Sequential(
#         #     nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, bias=False),
#         #     nn.GroupNorm(int(mid_channels / 16), mid_channels),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2, bias=False),
#         #     nn.GroupNorm(int(out_channels / 16), out_channels),
#         #     # nn.ReLU(inplace=True)
#         # )
#         # self.shortcut = nn.Sequential(
#         #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 1),
#         #     nn.GroupNorm(int(out_channels/16),out_channels),
#         # )
#         # self.ReLU = nn.ReLU()
#     def forward(self, x):
#         return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Max = nn.MaxPool2d(2)
        self.Avg = nn.AvgPool2d(2)
        self.DC=DoubleConv(in_channels, out_channels)
        self.D_DC=DoubleConv(in_channels*2, out_channels)

    def forward(self, x, fea, num):
        if num<3:
            temp = torch.cat([x,fea],dim=1)
            # temp = fea+x
            out = (0.85*self.Max(temp)+0.15*self.Avg(temp))
        else:
            out = 0.85 * self.Max(x) + 0.15 * self.Avg(x)
        # out =self.Max(x)
        if num<3:
            result = self.D_DC(out)
        else:
            result = self.DC(out)
        return result

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1)

    def forward(self, x):
        return self.conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(int((in_channels//2)/16), in_channels//2),
                nn.ReLU(inplace=True)
            )
            self.conv = DoubleConv(in_channels, out_channels)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(x1.shape, x2.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels+3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)
        # self.up4 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        #     nn.GroupNorm(int(128/16),128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
        #     # nn.GroupNorm(int(64/16),64),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False),
        # )
        self.outc = OutConv(64 , 3)
        # self.outc_four = OutConv(64 + 3 + 1, 3)
        # self.FeaEx = FeaNet(64+3+3+1)
        # self.decode = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        # self.log = nn.LogSoftmax(dim =1 )

    # def forward(self, x, Pre_fea, num, Fea):
    #     if num < 4:
    #         f = Pre_fea
    #     else:
    #         f = x
    #     x1 = self.inc(f)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)
    #     u = self.up1(x5, x4)
    #     u = self.up2(u, x3)
    #     u = self.up3(u, x2)
    #     u = self.up4(u, x1)
    #     u = self.outc(torch.cat([u,self.decode(x),Fea],dim=1))
    #     # logits = self.log(self.outc(x))
    #     return u
    def forward(self, x, Prex, PreFea, num):
        fea = []
        Re = Prex - x
        Re[Re<=0] = 0
        x1 = self.inc(torch.cat([x,Re],dim=1))
        fea.append(x1)
        x2 = self.down1(x1,PreFea[0],num)
        fea.append(x2)
        x3 = self.down2(x2,PreFea[1],num)
        fea.append(x3)
        x4 = self.down3(x3,PreFea[2],num)
        fea.append(x4)
        x5 = self.down4(x4,PreFea[3],num)
        fea.append(x5)
        u = self.up1(x5, x4)
        u = self.up2(u, x3)
        u = self.up3(u, x2)
        u = self.up4(u, x1)
        u = self.outc(u)
        # if num > 2
        #     u = self.outc(torch.cat([u,x],dim=1))
        # else
        #     u = self.outc(torch.cat([u,x,]))
        # logits = self.log(self.outc(x))


        return fea, u

class UNet_Four(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet_Four, self).__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor =  1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3)
        # self.outc_four = OutConv(64 + 3 + 1, 3)
        # self.decode = nn.Sequential(
        #     nn.Conv2d(3,64,kernel_size=3,padding=1,stride=1),
        #     nn.GroupNorm(8, 64),
        #     nn.ReLU(inplace=True),
        # )
        # self.FeaEx = FeaNet(64+3+1)
        # self.decode = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        # self.log = nn.LogSoftmax(dim =1 )

    # def forward(self, x, Pre_fea, num, Fea):
    #     if num < 4:
    #         f = Pre_fea
    #     else:
    #         f = x
    #     x1 = self.inc(f)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)
    #     u = self.up1(x5, x4)
    #     u = self.up2(u, x3)
    #     u = self.up3(u, x2)
    #     u = self.up4(u, x1)
    #     u = self.outc(torch.cat([u,self.decode(x),Fea],dim=1))
    #     # logits = self.log(self.outc(x))
    #     return u
    def forward(self, x, num):

        # Re = Prex - x
        # Re[Re<=0] = 0
        fea = []
        x1 = self.inc(x)
        fea.append(x1)
        x2 = self.down1(x1,[] , num)
        fea.append(x2)
        x3 = self.down2(x2, [], num)
        fea.append(x3)
        x4 = self.down3(x3, [], num)
        fea.append(x4)
        x5 = self.down4(x4, [], num)
        fea.append(x5)
        # print (x5.shape, x4.shape)
        u = self.up1(x5, x4)
        u = self.up2(u, x3)
        u = self.up3(u, x2)
        u = self.up4(u, x1)
        u = self.outc(u)
        # if num > 2
        #     u = self.outc(torch.cat([u,x],dim=1))
        # else
        #     u = self.outc(torch.cat([u,x,]))
        # logits = self.log(self.outc(x))
        # u = self.FeaEx(torch.cat([u, x, mask], dim=1))
        return fea, u

class Or_Unet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(Or_Unet, self).__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor =  1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3)
        # self.outc_four = OutConv(64 + 3 + 1, 3)
        # self.decode = nn.Sequential(
        #     nn.Conv2d(3,64,kernel_size=3,padding=1,stride=1),
        #     nn.GroupNorm(8, 64),
        #     nn.ReLU(inplace=True),
        # )
        # self.FeaEx = FeaNet(64+3+1)
        # self.decode = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        # self.log = nn.LogSoftmax(dim =1 )

    # def forward(self, x, Pre_fea, num, Fea):
    #     if num < 4:
    #         f = Pre_fea
    #     else:
    #         f = x
    #     x1 = self.inc(f)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)
    #     u = self.up1(x5, x4)
    #     u = self.up2(u, x3)
    #     u = self.up3(u, x2)
    #     u = self.up4(u, x1)
    #     u = self.outc(torch.cat([u,self.decode(x),Fea],dim=1))
    #     # logits = self.log(self.outc(x))
    #     return u
    def forward(self, x):

        # Re = Prex - x
        # Re[Re<=0] = 0
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print (x5.shape, x4.shape)
        u = self.up1(x5, x4)
        u = self.up2(u, x3)
        u = self.up3(u, x2)
        u = self.up4(u, x1)
        u = self.outc(u)
        # if num > 2
        #     u = self.outc(torch.cat([u,x],dim=1))
        # else
        #     u = self.outc(torch.cat([u,x,]))
        # logits = self.log(self.outc(x))
        # u = self.FeaEx(torch.cat([u, x, mask], dim=1))
        return u

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


