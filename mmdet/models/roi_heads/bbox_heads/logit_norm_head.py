import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np

from .bbox_head import BBoxHead
from .convfc_bbox_head import Shared2FCBBoxHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy


@HEADS.register_module()
class LogitNormHead(Shared2FCBBoxHead):

    def __init__(self, momentum=1e-4, *args, **kwargs):
        super(LogitNormHead, self).__init__(*args, **kwargs)
        self.bn = nn.BatchNorm1d(self.num_classes + 1, eps=1e-05, momentum=momentum, affine=False)
    
    def get_statistics(self):
        mean_val = self.bn.running_mean
        mean_val[-1] = 0
        std_val = torch.sqrt(torch.clamp(self.bn.running_var, min=1e-11))
        std_val[-1] = 1
        beta = torch.zeros_like(mean_val)
        beta[:-1] = mean_val[:-1].min()

        return mean_val.view(1, -1), std_val.view(1, -1), beta.view(1, -1)
    
    def forward(self, x):
        cls_score, bbox_pred = super(LogitNormHead, self).forward(x)

        if self.training:
            cls_score = self.bn(cls_score)
            return cls_score, bbox_pred
        else:
            mean_val, std_val, beta = self.get_statistics()
            cls_score = (cls_score - (mean_val - beta)) / std_val
            return cls_score, bbox_pred