#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import numpy as np
import torch.nn as nn

class SmoothL1Loss(nn.Module):
    def __init__(self, w=10, e=2, label_th=0.9, ada_pow=0):
        super(SmoothL1Loss, self).__init__()
        self.label_th = label_th
        self.ada_pow = ada_pow
        # self.L2 = torch.square()
    def forward(self, pred, target):
        points_num = int(target.shape[1]/3)
        pred = pred.view(-1, 2)
        target = target.view(-1, 3)
        keep_mask = target[:, 2] >= self.label_th
        weight = torch.pow(target[:, 2], self.ada_pow) * keep_mask
        diff = pred - target[:, :2]
        abs_diff = diff.abs()
        l2_flag = (abs_diff.data < 1).float()
        y = l2_flag *torch.square(abs_diff) + (1 - l2_flag) * abs_diff
        # return (y.sum(1)*weight).view(-1, points_num).sum(1)# 返回加权后每个obj的point损失
        return (y.sum(1) * weight).view(-1, points_num).mean(1) * 6  # 返回加权后每个obj的point损失

class WingLoss(nn.Module):
    def __init__(self, w=10, e=2, label_th=0.9, ada_pow=0):
        super(WingLoss, self).__init__()
        # https://arxiv.org/pdf/1711.06753v4.pdf   Figure 5
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)
        self.label_th = label_th
        self.ada_pow = ada_pow
    # def forward(self, pred, target, sigma=1):
    def forward(self, pred, target, sigma=1):
        points_num = int(target.shape[1]/3)
        pred = pred.view(-1, 2)
        target = target.view(-1, 3)
        keep_mask = target[:, 2] >= self.label_th
        weight = torch.pow(target[:, 2], self.ada_pow) * keep_mask
        diff = pred - target[:, :2]
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        # return (y.sum(1)*weight).view(-1, points_num).sum(1)# 返回加权后每个obj的point损失
        return (y.sum(1) * weight).view(-1, points_num).mean(1) * 6  # 返回加权后每个obj的point损失


class Varifocalloss(nn.Module):
    def __init__(self):
        super(Varifocalloss, self).__init__()
    def forward(self, pred, target, alpha=0.25, gamma=2):
        assert pred.shape[0] == target.shape[0]
        assert pred.max() <= 1 and pred.min() >= 0

        alpha_arr = torch.zeros_like(target)
        alpha_arr[target> 0] = 0.25
        alpha_arr[target == 0] = (1.-alpha)

        p_arr = torch.zeros_like(target)
        p_arr[target > 0] = -torch.pow(1-pred[target > 0], gamma) * torch.log(pred[target > 0])
        p_arr[target == 0] = -torch.pow(pred[target == 0], gamma) * torch.log(1-pred[target == 0])

        loss = alpha_arr*p_arr
        return loss

class Focalloss(nn.Module):
    def __init__(self):
        super(Focalloss, self).__init__()
    def forward(self, pred, target, alpha=0.25, gamma=2):
        assert pred.shape[0] == target.shape[0]
        assert pred.max() <= 1 and pred.min() >= 0

        alpha_arr = torch.zeros_like(target)
        alpha_arr[target>0] = 0.25
        alpha_arr[target == 0] = (1.-alpha)

        p_arr = torch.zeros_like(target)
        p_arr[target > 0] = -torch.pow(1-pred[target > 0], gamma) * torch.log(pred[target > 0])
        p_arr[target == 0] = -torch.pow(pred[target == 0], gamma) * torch.log(1-pred[target == 0])

        loss = alpha_arr*p_arr
        return loss


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="alpha_ciou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        '''cx,cy,w,h'''
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou
            # loss = 1 - iou*iou
            # return iou
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "ciou":
            epsilon = torch.tensor(1e-07)

            b1_xy = pred[..., :2]
            b1_wh = pred[..., 2:4]
            b1_wh_half = b1_wh / 2.
            b1_mins = b1_xy - b1_wh_half
            b1_maxes = b1_xy + b1_wh_half

            b2_xy = target[..., :2]
            b2_wh = target[..., 2:4]
            b2_wh_half = b2_wh / 2.
            b2_mins = b2_xy - b2_wh_half
            b2_maxes = b2_xy + b2_wh_half

            center_distance = torch.sum(torch.square(b1_xy - b2_xy), axis=-1)
            enclose_mins = torch.minimum(b1_mins, b2_mins)
            enclose_maxes = torch.maximum(b1_maxes, b2_maxes)
            enclose_wh = torch.maximum(enclose_maxes - enclose_mins, torch.tensor(0.0))
            # -----------------------------------------------------------#
            #   计算对角线距离
            #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
            # -----------------------------------------------------------#
            enclose_diagonal = torch.sum(torch.square(enclose_wh), axis=-1)

            R_d = 1.0 * (center_distance) / torch.maximum(enclose_diagonal,  epsilon)

            v = 4 * torch.square(torch.atan2(b1_wh[..., 0], torch.maximum(b1_wh[..., 1], epsilon))- torch.atan2(b2_wh[..., 0],torch.maximum(b2_wh[..., 1], epsilon)))/(torch.pi * torch.pi)
            alpha = v / torch.maximum((1.0 - iou + v), epsilon)

            # ciou = ciou - alpha * v

            loss = 1 - iou + R_d + alpha*v

        elif self.loss_type == "alpha_ciou":
            alpha_power = 2

            epsilon = torch.tensor(1e-07)

            b1_xy = pred[..., :2]
            b1_wh = pred[..., 2:4]
            b1_wh_half = b1_wh / 2.
            b1_mins = b1_xy - b1_wh_half
            b1_maxes = b1_xy + b1_wh_half

            b2_xy = target[..., :2]
            b2_wh = target[..., 2:4]
            b2_wh_half = b2_wh / 2.
            b2_mins = b2_xy - b2_wh_half
            b2_maxes = b2_xy + b2_wh_half

            center_distance = torch.sum(torch.square(b1_xy - b2_xy), axis=-1)
            enclose_mins = torch.minimum(b1_mins, b2_mins)
            enclose_maxes = torch.maximum(b1_maxes, b2_maxes)
            enclose_wh = torch.maximum(enclose_maxes - enclose_mins, torch.tensor(0.0))
            # -----------------------------------------------------------#
            #   计算对角线距离
            #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
            # -----------------------------------------------------------#
            enclose_diagonal = torch.sum(torch.square(enclose_wh), axis=-1)

            R_d = 1.0 * (center_distance) / torch.maximum(enclose_diagonal,  epsilon)

            v = 4 * torch.square(torch.atan2(b1_wh[..., 0], torch.maximum(b1_wh[..., 1], epsilon))- torch.atan2(b2_wh[..., 0],torch.maximum(b2_wh[..., 1], epsilon)))/(torch.pi * torch.pi)
            alpha = v / torch.maximum((1.0 - iou + v), epsilon)

            # ciou = ciou - alpha * v

            loss = 1 - torch.pow(iou, alpha_power) + torch.pow(R_d, alpha_power) +torch.pow(alpha*v, alpha_power)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

if __name__ == "__main__":
    import numpy as np

    '''focal loss test 
    where alpha=0.25 ,gamma =2
    the right output：
    https://blog.csdn.net/kill2013110/article/details/125569898
    '''
    pred = torch.tensor([0.9, 0.968, 0.1, 0.032, 0.1, 0.9], dtype=torch.float32)
    target = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32)
    fl = Focalloss()
    output = fl.forward(pred, target, .25, 2)



    '''iou loss test'''
    # pred = torch.tensor(np.load('pred_np.npy'))
    # target = torch.tensor(np.load('target_np.npy'))
    #
    # iou = IOUloss(reduction="none")
    # giou= IOUloss(loss_type="giou")
    # a_ciou=IOUloss(loss_type="alpha_ciou")
    #
    # b0 = torch.tensor([125, 125, 50, 50], dtype=torch.float32)
    # b1 = torch.tensor([65, 65, 30, 30], dtype=torch.float32)
    # b2 = torch.tensor([85, 115, 30, 30], dtype=torch.float32)
    # b3 = torch.tensor([115, 115, 30, 30], dtype=torch.float32)
    # b4 = torch.tensor([115, 125, 30, 30], dtype=torch.float32)
    # b5 = torch.tensor([125, 125, 18, 50], dtype=torch.float32)
    # b6 = torch.tensor([125, 125, 30, 30], dtype=torch.float32)
    #
    # x = [b1, b2,b3,b4, b5, b6]
    # for i in x:
    #     print('\n')
    #     print(iou.forward(i, b0))
    #     print(a_ciou.forward(i, b0))
    # print()
    #
    # print(iou.forward(pred, target))
    # print(a_ciou.forward(pred, target))
    # iou.forward(pred, target)


