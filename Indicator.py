import argparse

import numpy
import numpy as np
import math
import torch
import copy


from torch.autograd import Variable
from PIL import Image
from torch.utils.data import DataLoader

# import click
import numpy as np
from DataSet_Test import *
from Net_Tool_Sec import *
from Loss import *



# def psnr(img1, img2):
#     m1 = img1.numpy()
#     m2 = img2.numpy()
#     img1 = np.float64(m1)
#     img2 = np.float64(m2)
#     mse = numpy.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
#
#
# def ssim(img1, img2):
#
#     img1 = Variable(img1, requires_grad=False)  # torch.Size([256, 256, 3])
#     img2 = Variable(img2, requires_grad=False)
#     # print (img1.shape)
#     ssim_value = pytorch_ssim.ssim(img1, img2).item()
#     return ssim_value
#
# def get_psnr_ssim(original, contrast):
#
#     psnrValue = psnr(original, contrast)
#     ssimValue = ssim(original, contrast)
#     return  psnrValue, ssimValue


def psnr(gt, img):
    """
    calculate psnr between two images
    :param gt: groundtruth image
    :param img: inference image
    :return: psnr
    """
    mse = torch.mean( (gt - img) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# create gaussian kernel by multiply two gaussian distribution
# extend to 3 channel
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

class conv(nn.Module):
    def __init__(self):
        super(conv, self).__init__()
        # self.dir =  [(0,1),(0,-1),(1,0),(-1,0)]
        self.snow_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        # print (self.snow_conv.weight.shape)
        self.x = torch.Tensor([[1,1,1],[1,1,1],[1,1,1]])
        self.x1 = torch.stack((self.x, self.x, self.x),dim=0)
        self.weight = torch.unsqueeze(self.x1,dim=1)
        self.weight = self.weight.to(device='cuda:0')
        self.snow_weight = nn.Parameter(data=self.weight, requires_grad=False)
        self.snow_conv.weight = self.snow_weight
        self.max = nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.avg = nn.AvgPool2d(kernel_size=3,padding=1,stride=1)
        # torch.nn.init.constant_(self.snow_conv.weight, self.weight)
        # self.snow_conv.weight =
        for param in self.snow_conv.parameters():
            param.requires_grad = False




        self.mask_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        self.mask_conv.weight = self.snow_weight
        for param in self.mask_conv.parameters():
            param.requires_grad = False
        # self.Pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=1)
    def forward(self, snow, mask, i):

        max_get = self.max(snow)
        # avg_get = self.avg(snow)


        no_update = mask==0

        pre_mask = torch.ones_like(snow)
        pre_mask = pre_mask.masked_fill_(no_update ,0.0)
        # print (torch.count_nonzero(no_update_holes).item(),'@@@@@@@@@@@')
        # print (snow.dtype, mask.dtype)
        output = self.snow_conv(snow*mask)

        output_mask = self.mask_conv(mask*1.0)
        # save_image(output, "Padding_Test/re_%d.png" % i, nrow=1, normalize=False)
        # print (self.snow_conv.weight)
        update_holes = output_mask==0

        mask_sum = output_mask.masked_fill_(update_holes, 1.0)
        output = output/mask_sum

        output = output.masked_fill_(update_holes, 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(update_holes, 0.0)
        sum = torch.count_nonzero(update_holes).item()
        # return  new_mask, pre_mask, sum, max_get, avg_get
        return new_mask, pre_mask, sum, output, max_get
class partial_avg(nn.Module):
    def __init__(self):
        super(partial_avg, self).__init__()
        self.conv = conv()

    def forward(self, snow, mask):
        snow_copy = copy.deepcopy(snow)
        i = 0
        while 1:
            i+=1
            # mask = self.Pad(mask)
            new_mask, pre_mask, sum, out, max_get= self.conv(snow, mask, i)

            #**************************************************************
            temp = 0.6*max_get*(1-pre_mask) + 0.4*out*(1-pre_mask)
            # snow = snow*(~pre_mask) + pre_mask*output_snow
            # pat = (1-new_mask) * snow_copy
            # snow = snow*pre_mask + (1-pre_mask) * temp * new_mask + pat
            #***************************************************************


            snow =  snow*pre_mask + temp*new_mask
            mask = new_mask
            # save_image(snow, "Padding_Test/re_%d.png" % i, nrow=1, normalize=False)
            #print (sum)
            if sum == 0 :
                break
            # print(len(l), "!!!!!!!!!!!!!!!!!!!!!")
            # if len(l) == 0:
            #     break
        return snow
if __name__ == '__main__':



    argparser = argparse.ArgumentParser(description='Train the model')

    argparser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )

    argparser.add_argument(
        '-r',
        '--root',
        # default='C:/DataSet/t/all/all/cat_img',
        default = "C:/Liukun/selectImage",
        type=str,
        help='root directory of trainset'
    )

    argparser.add_argument(
        '-dir',
        type=str,
        default='weight/',
        help='path to store the model checkpoints'
    )

    argparser.add_argument(
        '-iter',
        '--iterations',
        type=int,
        default=2000
    )

    argparser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        default=1e-3
    )

    argparser.add_argument(
        '--batch_size',
        type=int,
        default=1
    )

    argparser.add_argument(
        '-beta',
        type=int,
        default=4,
        help='the scale of the pyramid maxout'
    )

    argparser.add_argument(
        '-gamma',
        type=int,
        default=4,
        help='the levels of the dilation pyramid'
    )

    argparser.add_argument(
        '--weight_decay',
        type=float,
        default=5e-4
    )

    argparser.add_argument(
        '--weight_mask',
        type=float,
        default=3,
        help='the weighting to leverage the importance of snow mask'
    )

    argparser.add_argument(
        '--save_schedule',
        type=int,
        nargs='+',
        default=[],
        help='the schedule to save the model'
    )

    argparser.add_argument(
        '--mode',
        type=str,
        default='original',
        help='the architectural mode of DesnowNet'
    )

    # argparser.add_argument('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')

    args = argparser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # net, starting_epoch = build_network(snapshot, backend)
    # data_path = os.path.abspath(os.path.expanduser(data_path))
    # models_path = os.path.abspath(os.path.expanduser(models_path))
    # os.makedirs(models_path, exist_ok=True)
    li_snow=['Snow100K-L','Snow100K-M','Snow100K-S']
    # li_snow = ['Snow100K-L']
    net = CSR_Net().to(device=args.device)
    check_path = 'Weight/checkpoints_G_Res_five_block_find_fault_itea63.pth'
    check_point = torch.load(check_path)
    net.load_state_dict(check_point['model_state_dict'])

    Seg_Mask_25 = AttU_Net().to(device=args.device)
    checkpoint = torch.load('Weight/checkpoints_Unet_seg_Zero_point_two_five_itea90.pth')
    Seg_Mask_25.load_state_dict(checkpoint['model_state_dict'])
    P_Avg = partial_avg().to(device=args.device)
    Max = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
    Avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
    for i in li_snow:
        path = args.root+'/'+i
        # gt_root = os.path.join(path, 'gt')
        # synthetic_root = os.path.join(path, 'synthetic')
        gt_root = path
        synthetic_root = path
        dataset = snow_dataset(gt_root, synthetic_root, mode='train')
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)


        iteration = 0
        number = 0
        sum = 0
        net.eval()
        # while iteration < args.iterations:
        #     iteration += 1
        number = 0
        #     sum = 0
        sum_psnr = 0
        sum_ssim = 0
        with torch.no_grad():
            for data in data_loader:
                number += 1
                if number > 100:
                    sum_psnr = sum_psnr*1.0/(number-1)
                    sum_ssim = sum_ssim*1.0/(number-1)
                    break;
                snow_img, gt_img = data
                snow_img = snow_img.to(device = args.device)
                gt_img = gt_img.to(device = args.device)
                con_snow = 1 - snow_img
                con_gt = 1 - gt_img

                # gt_fir_img = 0.85 * Max(con_gt) + 0.15 * Avg(con_gt)
                # gt_sec_img = 0.85 * Max(gt_fir_img) + 0.15 * Avg(gt_fir_img)
                # gt_thr_img = 0.85 * Max(gt_sec_img) + 0.15 * Avg(gt_sec_img)
                # gt_fou_img = 0.85*Max(gt_thr_img)+0.15*Avg(gt_thr_img)

                snow_fir_img = 0.85 * Max(con_snow) + 0.15 * Avg(con_snow)
                snow_sec_img = 0.85 * Max(snow_fir_img) + 0.15 * Avg(snow_fir_img)
                snow_thr_img = 0.85 * Max(snow_sec_img) + 0.15 * Avg(snow_sec_img)

                out = Seg_Mask_25(snow_img)
                Mask = torch.argmax(out, dim=1, keepdim=True)
                Mask = torch.cat([Mask, Mask, Mask], dim=1)

                Recover_snow_thr_img = P_Avg(snow_thr_img, 1 - Mask)
                A_fir, A_sec, A_thr, A = net(snow_img,con_snow,snow_fir_img, snow_sec_img, Recover_snow_thr_img)

                mer = torch.cat([snow_img, con_snow, gt_img, con_gt, A_fir, A_sec, A_thr, A, 1-A], dim = 3)
                snow_sa = snow_img
                A_sa = 1-A
                # save_image(A_sa, "select_show_image/" + i + "_result_%d.png" % number, nrow=1, normalize=False)
                # save_image(snow_sa, "select_show_image/" + i + "_original_%d.png" % number, nrow=1, normalize=False)


                sum_psnr += psnr(gt_img,1-A)
                sum_ssim += ssim(gt_img,1-A)





