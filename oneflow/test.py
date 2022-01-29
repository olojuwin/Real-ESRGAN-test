import math
import oneflow
from oneflow import nn as nn
from oneflow.nn import functional as F
from oneflow.nn import init as init
from oneflow.nn.modules.batchnorm import _BatchNorm
from rrdbnet_arch import *
import os
import numpy as np
import cv2
import time
from graph import  EvalGraph

class RealESRNet(object):
    def __init__(self, base_dir='./', model=None, scale=2, graph=False, fp16=False):
        self.base_dir = base_dir
        self.scale = scale
        self.graph = graph
        self.fp16 = fp16
        self.load_srmodel(base_dir, model)



    def load_srmodel(self, base_dir, model):
        self.srmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=self.scale)

        loadnet = oneflow.load(os.path.join(
            self.base_dir, 'weights', 'RealESRGAN_x2plus'))

        self.srmodel.load_state_dict(loadnet, strict=True)
        self.srmodel = self.srmodel.cuda()
        self.srmodel.eval()
        if self.graph:
            self.srmodel = EvalGraph(self.srmodel, fp16=False)
        

    @oneflow.no_grad()
    def process(self, img):
        img = img.astype(np.float32) / 255.
        img = oneflow.tensor(np.transpose(
            img[:, :, [2, 1, 0]], (2, 0, 1)), dtype=oneflow.float)
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

        with oneflow.no_grad():
            output = self.srmodel(img)
        # remove extra pad
        if mod_scale is not None:
            _, _, h, w = output.size()
            output = output[:, :, 0:h - h_pad, 0:w - w_pad]
        output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        return output


SR_net = RealESRNet(graph=True,fp16=True)
im_lr = cv2.imread("img/test_1.jpg")
img_sr = SR_net.process(im_lr)
start_t=time.time()
for i in range(20):
    img_sr=SR_net.process(im_lr)
print((time.time()-start_t)/20)
cv2.imwrite("4_sr.jpg", img_sr)
