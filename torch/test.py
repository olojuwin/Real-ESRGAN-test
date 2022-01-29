import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from rrdbnet_arch import *
import os
import numpy as np
import cv2
import time
# model = RRDBNet(3, 3, 64, 23, 32, 2)
# loadnet = torch.load("./RealESRGAN_x2plus")
# if 'params_ema' in loadnet:
#     keyname = 'params_ema'
# else:
#     keyname = 'params'
# model.load_state_dict(loadnet[keyname], strict=True)
# model.eval()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# if False:
#     model = model.half()


class RealESRNet(object):
    def __init__(self, base_dir='./', model=None, scale=2):
        self.base_dir = base_dir
        self.scale = scale
        self.load_srmodel(base_dir, model)

    def load_srmodel(self, base_dir, model):
        self.srmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=self.scale)
        if model is None:
            loadnet = torch.load(os.path.join(
                self.base_dir, 'weights', 'RealESRGAN_x2plus.pth'))
        else:
            loadnet = torch.load(os.path.join(
                self.base_dir, 'weights', model+'.pth'))
        self.srmodel.load_state_dict(loadnet['params_ema'], strict=True)
        self.srmodel.eval()
        self.srmodel = self.srmodel.cuda()

    @torch.no_grad()
    def process(self, img):
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(
            img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).cuda()

        if self.scale == 2:
            mod_scale = 2
        elif self.scale == 1:
            mod_scale = 4
        else:
            mod_scale = None
        if mod_scale is not None:
            h_pad, w_pad = 0, 0
            _, _, h, w = img.size()
            if (h % mod_scale != 0):
                h_pad = (mod_scale - h % mod_scale)
            if (w % mod_scale != 0):
                w_pad = (mod_scale - w % mod_scale)
            img = F.pad(img, (0, w_pad, 0, h_pad), 'reflect')

        try:
            with torch.no_grad():
                output = self.srmodel(img)
            # remove extra pad
            if mod_scale is not None:
                _, _, h, w = output.size()
                output = output[:, :, 0:h - h_pad, 0:w - w_pad]
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)

            return output
        except:
            return None


SR_net = RealESRNet()
im_lr = cv2.imread("img/test_1.jpg")
img_sr = SR_net.process(im_lr)
start_t = time.time()
# for i in range(20):
#     img_sr = SR_net.process(im_lr)
print((time.time()-start_t)/20)
cv2.imwrite("4_sr.jpg", img_sr)
