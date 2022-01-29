from email.mime import base
import torch
import oneflow as flow
from rrdbnet_arch import *
import os

scale = 2
base_dir = "./"
srmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                  num_block=23, num_grow_ch=32, scale=scale)

parameters = torch.load(os.path.join(
    base_dir, 'weights', 'RealESRGAN_x2plus.pth'))['params_ema']


new_parameters = dict()
for key, value in parameters.items():
    if "num_batches_tracked" not in key:
        val = value.detach().cpu().numpy()
        new_parameters[key] = val
srmodel.load_state_dict(new_parameters)
flow.save(srmodel.state_dict(), "weight/RealESRGAN_x2plus")
