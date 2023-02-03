import os
import torch
import argparse
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
import copy

# import click
import numpy as np
from DataSet import *
from Net_Tool_Sec import *

from Unet_Tool import *
import opt
from Loss import *
# def conv(snow, mask, li, h, w):
#     l = []
#     mask_img = copy.deepcopy(mask)
#     snow_img = copy.deepcopy(snow)
#     for e in li:
#         i, j, k, m = e[0], e[1], e[2], e[3]
#         num = 0
#         avg = 0
#         # print(i, j, k, m)
#         for n in dir:
#            if k+n[0]<h and k+n[0]>=0 and m+n[1]<w and m+n[1]>=0 and mask_img[i,j,k+n[0],m+n[1]] < 0.5:
#                # print ("!!!!!")
#                num += 1
#                avg += snow_img[i,j,k+n[0],m+n[1]]
#                # avg = max(avg,snow_img[i,j,k+n[0],m+n[1]])
#         if num > 0:
#             snow [i,j,k,m] = avg*1.0/(num*1.0)
#             mask [i,j,k,m] = 0
#         else:
#             l.append((i,j,k,m))
#
#     return snow, mask, l
#
#
# def partial_avg(snow, mask):
#
#      b, c, h, w = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
#      li = []
#      li = torch.nonzero(mask>=0.5)
#      print (len(li))
#      while 1:
#         snow, mask, l = conv(snow,  mask,  li, h, w)
#         print (len(l),"!!!!!!!!!!!!!!!!!!!!!")
#         if len(l) == 0:
#             break
#      return snow


#
# class conv(nn.Module):
#     def __init__(self):
#         super(conv, self).__init__()
#         # self.dir =  [(0,1),(0,-1),(1,0),(-1,0)]
#         self.snow_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
#         # print (self.snow_conv.weight.shape)
#         self.x = torch.Tensor([[0,1,0],[1,0,1],[0,1,0]])
#         self.x1 = torch.stack((self.x, self.x, self.x),dim=0)
#         self.weight = torch.unsqueeze(self.x1,dim=1)
#         self.weight = self.weight.to(device='cuda:0')
#         self.snow_weight = nn.Parameter(data=self.weight, requires_grad=False)
#         self.snow_conv.weight = self.snow_weight
#         # torch.nn.init.constant_(self.snow_conv.weight, self.weight)
#         # self.snow_conv.weight =
#         for param in self.snow_conv.parameters():
#             param.requires_grad = False
#
#
#
#
#         self.mask_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
#         self.mask_conv.weight = self.snow_weight
#         for param in self.mask_conv.parameters():
#             param.requires_grad = False
#         # self.Pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=1)
#     def forward(self, snow, mask, i):
#
#         no_update_holes = mask==0
#         # print (torch.count_nonzero(no_update_holes).item(),'@@@@@@@@@@@')
#
#         output = self.snow_conv(snow*mask)
#
#         output_mask = self.mask_conv(mask)
#         # save_image(output, "Padding_Test/re_%d.png" % i, nrow=1, normalize=False)
#         # print (self.snow_conv.weight)
#         update_holes = output_mask == 0
#         mask_sum = output_mask.masked_fill_(update_holes, 1.0)
#         output = output/mask_sum
#
#         # output = output.masked_fill_(update_holes, 0.0)
#         new_mask = torch.ones_like(output)
#         new_mask = new_mask.masked_fill_(update_holes, 0.0)
#         sum = torch.count_nonzero(update_holes).item()
#         return output, new_mask, no_update_holes, sum
#         # l = []
#         # mask_img = copy.deepcopy(mask)
#         # snow_img = copy.deepcopy(snow)
#         # for e in li:
#         #     i, j, k, m = e[0], e[1], e[2], e[3]
#         #     num = 0
#         #     avg = 0
#         #     # print(i, j, k, m)
#         #     for n in self.dir:
#         #         if k + n[0] < h and k + n[0] >= 0 and m + n[1] < w and m + n[1] >= 0 and mask_img[i, j, k + n[0], m + n[1]] < 0.5:
#         #             # print ("!!!!!")
#         #             num += 1
#         #             avg += snow_img[i, j, k + n[0], m + n[1]]
#         #             # avg = max(avg,snow_img[i,j,k+n[0],m+n[1]])
#         #     if num > 0:
#         #         snow[i, j, k, m] = avg * 1.0 / (num * 1.0)
#         #         mask[i, j, k, m] = 0
#         #     else:
#         #         l.append((i, j, k, m))
#         #
#         # return snow, mask, l
#
# class partial_avg(nn.Module):
#     def __init__(self):
#         super(partial_avg, self).__init__()
#         self.conv = conv()
#
#     def forward(self, snow, mask):
#         # b, c, h, w = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
#         # num = b*c*h*w
#         # li = []
#         # li = torch.nonzero(mask >= 0.5)
#         # print(len(li))
#         i = 0
#         while 1:
#             i+=1
#             # mask = self.Pad(mask)
#             output_snow, new_mask, pre_mask, sum= self.conv(snow, mask, i)
#             snow = snow*(~pre_mask) + pre_mask*output_snow
#             mask = new_mask
#             # save_image(snow, "Padding_Test/re_%d.png" % i, nrow=1, normalize=False)
#             # print (sum)
#             if sum == 0 :
#                 break
#             # print(len(l), "!!!!!!!!!!!!!!!!!!!!!")
#             # if len(l) == 0:
#             #     break
#         return snow
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
        # snow_copy = copy.deepcopy(snow)
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
    # 设置随机数种子

    argparser = argparse.ArgumentParser(description='Train the model')

    argparser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )

    argparser.add_argument(
        '-r',
        '--root',
        default='C:/DataSet/t/all/all',
        type=str,
        help='root directory of trainset'
    )

    argparser.add_argument(
        '-dir',
        type=str,
        default='Weight/',
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
        default=5e-4
    )

    argparser.add_argument(
        '--batch_size',
        type=int,
        default=4
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
        default=1e-4
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
    gt_root = os.path.join(args.root, 'gt')
    mask_root = os.path.join(args.root, 'mask')
    synthetic_root = os.path.join(args.root, 'synthetic')
    dataset = snow_dataset(gt_root, mask_root, synthetic_root, mode='train')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=6,
                                              pin_memory=True)


    G_net = CSR_Net().to(device=args.device)


    G_optimizer = optim.Adam(G_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # D_optimizer = optim.Adam(D_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    Seg_Mask_25 = AttU_Net().to(device = args.device)
    checkpoint = torch.load(os.path.join(args.dir, 'checkpoints_Unet_seg_Zero_point_two_five_itea90.pth'), map_location='cpu')
    Seg_Mask_25.load_state_dict(checkpoint['model_state_dict'])


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, mode='min', factor=0.8, patience=8, verbose=True)
    seg_criterion = nn.CrossEntropyLoss()

    crit = nn.L1Loss()
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device=args.device)
    # crit_sum = nn.L1Loss(reduction='sum')
    P_Avg = partial_avg().to(device=args.device)
    Max = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
    Avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)

    min_num = 1e9+7
    it_min = 0
    G_net.train()
    # D_net.train()
    iteration = 0
    number = 0
    sum = 0
    while iteration < args.iterations:
        iteration += 1
        number = 0
        sum = 0
        for data in data_loader:
            number += 1
            G_optimizer.zero_grad()

            snow_img, gt_img, mask_h, mask_l= data
            # mask_h = torch.unsqueeze(mask_h, dim=1)
            # mask_h = torch.cat([mask_h, mask_h, mask_h],dim=1)
            gt_img = gt_img.to(device=args.device)
            snow_img = snow_img.to(device=args.device)
            mask_h = mask_h.to(device=args.device)
            # mask_l = torch.cat([mask_l,mask_l,mask_l],dim=1)
            mask_l = mask_l.to(device=args.device)

            Re_img = snow_img - gt_img
            con_snow = 1-snow_img
            con_gt = 1-gt_img

            gt_fir_img = 0.85*Max(con_gt)+0.15*Avg(con_gt)
            gt_sec_img = 0.85*Max(gt_fir_img)+0.15*Avg(gt_fir_img)
            gt_thr_img = 0.85*Max(gt_sec_img)+0.15*Avg(gt_sec_img)
            # gt_fou_img = 0.85*Max(gt_thr_img)+0.15*Avg(gt_thr_img)


            snow_fir_img = 0.85*Max(con_snow)+0.15*Avg(con_snow)
            snow_sec_img = 0.85*Max(snow_fir_img)+0.15*Avg(snow_fir_img)
            snow_thr_img = 0.85*Max(snow_sec_img)+0.15*Avg(snow_sec_img)
            # snow_fou_img = 0.85*Max(snow_thr_img)+0.15*Avg(snow_thr_img)

            # Re_img_fir = snow_fir_img - gt_fir_img

            # Light_Mask, A_Mask, A_fir, A_sec, A_thr, A_fou, out = net(snow_img,con_snow,snow_fir_img, snow_sec_img,snow_thr_img,snow_fou_img)
            # snow_img = P_Avg(snow_img,mask_h)
            # print ("!!!!!!!!!")
            # H_M = torch.cat([Heavy_Mask, Heavy_Mask, Heavy_Mask], dim=1)
            with torch.no_grad():
                out = Seg_Mask_25(snow_img)
                Mask = torch.argmax(out, dim=1, keepdim=True)
                Mask = torch.cat([Mask, Mask, Mask], dim=1)
                # Mask_more = Max(Mask.float())
                # out_more = Seg_Mask_17(snow_img)
                # Mask_more = torch.argmax(out_more, dim=1, keepdim=True)
                # Mask_more = torch.cat([Mask_more, Mask_more, Mask_more], dim=1)
                # Mask_more = Mask_more + Mask
                # Mask_more[Mask_more>=1] = 1
                Mask_more = Mask


                Recover_snow_thr_img = P_Avg(snow_thr_img, 1 - Mask)

            # print(out.shape)




            A_fir, A_sec, A_thr, A = G_net(snow_img, con_snow, snow_fir_img, snow_sec_img, Recover_snow_thr_img)
            # print (Re_A.shape, Mask_more.shape)
            # comp_img = (1 - Mask_more) * con_gt + Mask_more * Re_A
            #
            gen_loss = 0
            loss_dict = criterion(con_snow, 1-Mask, A, con_gt)

            hole_loss_a = crit(A * Mask_more, con_gt * Mask_more)
            # hole_loss_aa = crit(A_sec * Mask_more, con_gt * Mask_more)
            # hole_loss_aaa  = crit(A_thr * Mask_more, con_gt * Mask_more)
            hole_loss_fir = crit(A_fir * Mask_more, gt_fir_img * Mask_more)
            hole_loss_sec = crit(A_sec * Mask, gt_sec_img * Mask)
            hole_loss_thr = crit(A_thr * Mask, gt_thr_img * Mask)
            # hole_loss_fou = crit(A_fou * Mask, gt_fou_img * Mask) / torch.mean(Mask + 1e-8)
            gen_loss += 3 * (hole_loss_a  + hole_loss_fir + hole_loss_sec + hole_loss_thr)

            # re_valid_loss = crit(Re_A * (1 - Mask_more), con_gt * (1 - Mask_more))
            valid_loss_a = crit(A * (1 - Mask_more), con_gt * (1 - Mask_more))
            # valid_loss_aa = crit(A_sec * (1 - Mask_more), con_gt * (1 - Mask_more))
            # valid_loss_aaa = crit(A_thr * (1 - Mask_more), con_gt * (1 - Mask_more))
            valid_loss_fir = crit(A_fir * (1 - Mask_more), gt_fir_img * (1 - Mask_more))
            valid_loss_sec = crit(A_sec * (1 - Mask), gt_sec_img * (1 - Mask))
            valid_loss_thr = crit(A_thr * (1-Mask), gt_thr_img * (1-Mask))
            # valid_loss_fou = crit(A_fou * (1-Mask), gt_fou_img * (1-Mask)) / torch.mean((1-Mask) + 1e-8)
            gen_loss += (valid_loss_a  + valid_loss_fir + valid_loss_sec + valid_loss_thr)
            for key, coef in opt.LAMBDA_DICT.items():
                value = coef * loss_dict[key]
                print (key,":",value.item(),end=' ')
                gen_loss += value

            # sec_mask_loss = crit(sec_mask, gt_sec_img-snow_sec_img)
            # fir_mask_loss = crit(fir_mask, gt_fir_img-snow_fir_img)
            # a_fir_mask_loss = crit(A_fir_Mask, con_gt - con_snow)
            # a_sec_mask_loss = crit(A_sec_Mask, con_gt - con_snow)
            # a_thr_mask_loss = crit(A_thr_Mask, con_gt - con_snow)

            # gen_loss += 3*(sec_mask_loss + fir_mask_loss + a_fir_mask_loss)

            # generator backward
            # G_optimizer.zero_grad()
            gen_loss.backward()
            G_optimizer.step()

            sum = sum + gen_loss.item()

            # print ('iteration:',iteration,'loss1:',loss1.item(),'loss2:',loss2.item(),'loss4:',loss4.item(),'loss5:',loss5.item(), 'loss6:', loss6.item())
            print('iteration:', iteration,'hole_loss_a:', hole_loss_a.item(), 'hole_loss_fir:', hole_loss_fir.item(),'hole_loss_sec:', hole_loss_sec.item(),'hole_loss_thr:', hole_loss_thr.item())
            print('valid_loss_a:', valid_loss_a.item(), 'valid_loss_fir:', valid_loss_fir.item(), 'valid_loss_sec:', valid_loss_sec.item(), 'valid_loss_thr:', valid_loss_thr.item())
            # print('fir_mask_loss', fir_mask_loss.item(), 'sec_mask_loss', sec_mask_loss.item(),'a_fir_mask_loss', a_fir_mask_loss.item(),'a_sec_mask_loss')
            print ('\n')
        sum = sum * 1.0 / number
        print('iteration:', iteration, 'loss:', sum,"**********************************************************************************************************************************************************************************************************************************************************")

        scheduler.step(sum)
        if (iteration % 3 == 0):
            if (sum < min_num):
                it_min = iteration
                min_num = sum
                checkpoint = {
                'iteration': iteration,
                'model_state_dict': G_net.state_dict(),
                'optimizer_state_dict': G_optimizer.state_dict(),
                'min_iteration':it_min
            }
            torch.save(checkpoint, os.path.join(args.dir, 'checkpoints_G_Res_five_block_find_fault_itea{}.pth'.format(iteration)))
            # check = {
            #     'iteration': iteration,
            #     'model_state_dict': D_net.state_dict(),
            #     'optimizer_state_dict': D_optimizer.state_dict(),
            #     'min_iteration': it_min
            # }
            # torch.save(checkpoint, os.path.join(args.dir, 'checkpoints_D_itea{}.pth'.format(iteration)))
        # train_loss = np.mean(epoch_losses)



