import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class SSDHead(nn.Module):

    def __init__(self,
                 num_classes=81,
                 in_channels=[256,256,256,256,256],
                 aspect_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2])):
        super(SSDHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        num_anchors = [len(ratios) * 2 + 2 for ratios in aspect_ratios]
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * num_classes,
                    kernel_size=3,
                    padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            #[num_featuremap,w,h,c]
            # 计算分类损失 #permute(多维数组,[维数的组合])
            # 该函数是改变维数 #contiguous：view只能用在contiguous的variable上。
            # #如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
            # 返回rpn网络判断的anchor前后景分数
            cls_scores.append(cls_conv(feat).permute(0, 2, 3, 1).contiguous())
            bbox_preds.append(reg_conv(feat).permute(0, 2, 3, 1).contiguous())
        
        return cls_scores, bbox_preds