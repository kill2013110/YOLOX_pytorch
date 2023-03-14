#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from yolox.utils import bboxes_iou, meshgrid

from .losses import IOUloss, WingLoss, SmoothL1Loss, VariFocalLoss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        get_face_pionts=False,
        label_th=0.9,
        ada_pow=0,
        points_loss='Wing',
        points_loss_weight=0.,
        var_config=[None,None],
        reg_iou=True,
        box_loss='GIoU',
        box_loss_weight=5.,
        # cls_loss='BCE',
        cls_loss_weight=1,
        vari_dconv_mask=False,
        Assigner='SimOTA',
        TAL_alpha_beta_topk_eps=(1., 6., 13, 1e-09),
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.reg_iou = reg_iou
        self.var_config = var_config
        self.vari_dconv_mask = vari_dconv_mask
        if self.var_config[0] != None and self.vari_dconv_mask == True:
            self.dconv_mask = torch.nn.Parameter(torch.FloatTensor(torch.zeros([1, 9, 1, 1])), requires_grad=False)
            # self.dconv_mask[0, 4, 0, 0] = 1
            self.dconv_mask.requires_grad = True
            logger.info('dconv mask init state:')
            logger.info(self.dconv_mask)
        else: self.dconv_mask = None

        self.points_loss_weight = points_loss_weight
        self.get_face_pionts = get_face_pionts
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()

        self.reg_preds = nn.ModuleList()
        if self.reg_iou:
            self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3, stride=1, act=act,),
                        Conv(in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3, stride=1, act=act,),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3, stride=1, act=act,),
                        Conv(in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3, stride=1, act=act,),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=3 if 'last' == self.var_config[1] else 1,
                    stride=1,
                    padding=0,
                    # bias=False,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4+2*self.get_face_pionts,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            if self.reg_iou:
                self.obj_preds.append(
                    nn.Conv2d(
                        in_channels=int(256 * width),
                        out_channels=self.n_anchors * 1,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )

        self.use_l1 = False
        assert Assigner in ['SimOTA', 'TAL']
        if Assigner == 'SimOTA':
            self.get_assignments = self.SimOTA
        elif Assigner == 'TAL':
            self.TAL_alpha, self.TAL_beta, self.TAL_topk, self.TAL_eps = TAL_alpha_beta_topk_eps
            self.get_assignments = self.TaskAlignedAssigner

        if reg_iou:
            self.obj_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.cls_loss_fn = VariFocalLoss()
        if box_loss =='a_CIoU':
            self.iou_loss_fn = IOUloss(reduction="none", loss_type="alpha_ciou")
        elif box_loss =='GIoU':
            self.iou_loss_fn = IOUloss(reduction="none", loss_type="giou")
        elif box_loss =='IoU':
            self.iou_loss_fn = IOUloss(reduction="none", loss_type="iou")

        if self.get_face_pionts != 0:
            assert points_loss in ['SmoothL1', 'Wing']
            if points_loss == 'Wing':
                self.points_loss_fn = WingLoss(label_th=label_th, ada_pow=ada_pow)
            if points_loss == 'SmoothL1':
                self.points_loss_fn = SmoothL1Loss(label_th=label_th, ada_pow=ada_pow)



        self.l1_loss = nn.L1Loss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            if conv.bias != None:
                b = conv.bias.view(self.n_anchors, -1)
                b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        if self.reg_iou:
            for conv in self.obj_preds:
                b = conv.bias.view(self.n_anchors, -1)
                b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        # assert self.get_face_pionts == labels.shape
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            if self.reg_iou:
                obj_output = self.obj_preds[k](reg_feat)
            else:
                obj_output = torch.ones([reg_output.shape[0], 1, *reg_output.shape[2:]]).to(reg_output.device)

            if self.var_config[0] == None:
                ''' YOLOX origin Conv style:  '''
                cls_feat = cls_conv(cls_x)
                cls_output = self.cls_preds[k](cls_feat)
            else:
                #######################计算偏移量############################
                if '0offset' in self.var_config:
                    mnwh = reg_output[:, :4].clone()  # 注意m,n指的是第m行，第n列的特征点的位置,那该特征点位置为(n, m)
                    mnwh[:, 2:4] = torch.exp(mnwh[:, 2:4])
                    x, y, w, h = torch.chunk(mnwh, 4, dim=1)  # xc, yc的偏移量以及w和h
                    offset = torch.zeros([mnwh.shape[0], 18, *mnwh.shape[2:]], device=mnwh.device)
                    # 无位移的3*3 Dconv
                    '''经此对比试验，确认每个点的偏移量不是相对中心点'''
                    # 第一列:x-1  第三列:x+1
                    offset[:, 0::6] = 0-1
                    offset[:, 4::6] = 0+1

                    # 第一行:y-1 第三行:y+1
                    offset[:, 1:6:2] = 0-1
                    offset[:, 13:18:2] = 0+1  # 也可写作 offset[:, 13::2] = y+h/2

                elif '8points' in self.var_config:  # 8个关键点 + 特征点
                    assert points_output.shape[1] == 8 * 2
                    ''' 8points Dconv style
                    8points_output:
                        'nose_l','nose_r',  'mouth_l','mouth_r', 'eye_l','eye_r', 'mouth_t', 'mouth_b'
                    3*3 dconv:
                        e_l m_t e_r
                        n_l     n_r
                        m_l m_b m_r
                    3*3 dconv 8points_output index
                        7,8  12,13  10,11
                        0,1         2,3
                        4,5  14,15  6,7
                    '''
                    xy_points = points_output.clone()
                    offset = torch.zeros([xy_points.shape[0], 3 * 3 * 2, *xy_points.shape[2:]], device=xy_points.device)

                    # 无位移的3*3 Dconv
                    '''经此对比试验，确认每个点的偏移量不是相对中心点'''
                    # # 第一列:x-1  第三列:x+1
                    # offset[:, 0::6] = 0-1
                    # offset[:, 4::6] = 0+1
                    #
                    # # 第一行:y-1 第三行:y+1
                    # offset[:, 1:6:2] = 0-1
                    # offset[:, 13:18:2] = 0+1 # 也可写作 offset[:, 13::2] = y+h/2

                    offset[:, 0:8] = xy_points[:, [8, 9, 12, 13, 10, 11, 0, 1]]  #
                    offset[:, 10:] = xy_points[:, [2, 3, 4, 5, 14, 15, 6, 7]]

                    # 第一列:x- w/2  第三列:x+ w/2
                    offset[:, 0::6] += 1
                    offset[:, 4::6] -= 1

                    # 第一行:y- h/2 第三行:y+ h/2
                    offset[:, 1:6:2] += 1
                    offset[:, 13:18:2] -= 1  # 也可写作 offset[:, 13::2] = y+h/2-1

                elif 'star' in self.var_config:  # VFNet论文中
                    ''' Star Dconv style:   '''
                    mnwh = reg_output[:, :4].clone()  # 注意m,n指的是第m行，第n列的特征点的位置,那该特征点位置为(n, m)
                    mnwh[:, 2:4] = torch.exp(mnwh[:, 2:4])
                    x, y, w, h = torch.chunk(mnwh, 4, dim=1)  # xc, yc的偏移量以及w和h
                    offset = torch.zeros([mnwh.shape[0], 18, *mnwh.shape[2:]], device=mnwh.device)

                    # 第一列:x- w/2  第三列:x+ w/2
                    offset[:, 0::6] = x - w / 2 + 1
                    offset[:, 4::6] = x + w / 2 - 1

                    # 第一行:y- h/2 第三行:y+ h/2
                    offset[:, 1:6:2] = y - h / 2 + 1
                    offset[:, 13:18:2] = y + h / 2 - 1  # 也可写作 offset[:, 13::2] = y+h/2-1

                elif 'star_inter' in self.var_config:  # 完全插值
                    ''' Star Dconv style:   '''
                    mnwh = reg_output[:,:4].clone()  # 注意m,n指的是第m行，第n列的特征点的位置,那该特征点位置为(n, m)
                    mnwh[:, 2:4] = torch.exp(mnwh[:, 2:4])
                    x, y, w, h = torch.chunk(mnwh, 4, dim=1)  # xc, yc的偏移量以及w和h
                    offset = torch.zeros([mnwh.shape[0], 18, *mnwh.shape[2:]], device=mnwh.device)
                    # 十字线位置的x，y
                    offset[:, 2::6] = x
                    offset[:, 7:12:2] = y

                    # 第一列:x- w/2  第三列:x+ w/2
                    offset[:, 0::6] = x - w / 2 + 1
                    offset[:, 4::6] = x + w / 2 - 1

                    # 第一行:y- h/2 第三行:y+ h/2
                    offset[:, 1:6:2] = y - h / 2 + 1
                    offset[:, 13:18:2] = y + h / 2 - 1  # 也可写作 offset[:, 13::2] = y+h/2-1
                    '''
                    debug时的可视化代码：
                    import cv2
                    import numpy as np
                    org_img = cv2.imread('1.jpg',1)
                    img = org_img.copy()
                    box = mnwh[0,:,10,19].cpu().numpy().copy()
                    # print(box) 
                    box[0] = (19+box[0])*8
                    box[1] = (10+box[1])*8
                    box[2:] = box[2:]*8
                    # print(box)
                    cv2.rectangle(img, (int(box[0] - box[2] / 2),int(box[1] - box[3] / 2)),
                                      (int(box[0] + box[2] / 2),int(box[1] + box[3] / 2)),
                                  [255, 255, 0])
                    # x = img.copy()
                    c = offset[0,:,10,19]
                    print(c)
                    c = c.reshape([9,2]).cpu().numpy().copy()
                    print(c)
                    # # cv2.circle(x, i[::-1], 2, color=(255, 0, 0))
                    c[:,0] = (19+c[:,0])*8
                    c[:,1] = (10+c[:,1])*8
                    points = np.int16(c)
                    for i in points:
                        cv2.circle(img, i, 3, color=(0, 255, 255))
                    cv2.imshow('1',img)
                    cv2.waitKey()
                    '''
                elif 'dense star' in self.var_config:
                    ''' Dense Star Dconv style:   '''
                    assert self.var_config != 'dense star'
                    pass
                ###############################################################

                #######################根据偏移量进行可变形卷积###################
                if self.vari_dconv_mask: mask = self.dconv_mask.repeat([cls_x.shape[0], 1, *cls_x.shape[-2:]])
                else: mask = self.dconv_mask
                if 'early' in self.var_config:
                    Star_Dconv_out = deform_conv2d(cls_x, offset, cls_conv[0].conv.weight, padding=1, mask=mask)
                    cls_feat = cls_conv[1](cls_conv[0].act(cls_conv[0].bn(Star_Dconv_out)))
                    cls_output = self.cls_preds[k](cls_feat)
                elif 'late' in self.var_config:
                    cls_feat_early = cls_conv[0](cls_x)
                    Star_Dconv_out = deform_conv2d(cls_feat_early, offset, cls_conv[1].conv.weight, padding=1, mask=mask)
                    cls_feat = cls_conv[1].act(cls_conv[1].bn(Star_Dconv_out))
                    cls_output = self.cls_preds[k](cls_feat)
                elif 'last' in self.var_config:
                    cls_feat = cls_conv(cls_x)
                    cls_output = deform_conv2d(cls_feat, offset, self.cls_preds[k].weight, padding=1, mask=mask)
                ############################################################

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output[:, :4].view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid() if self.reg_iou else torch.ones(obj_output.shape).to(obj_output.device),
                    cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 4+ 2*self.get_face_pionts + 1 + self.num_classes if self.get_face_pionts else 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        if self.get_face_pionts:#outputs: [xc,yc,w,h, [x,y]*6 ,obj, cls...]
            for i in range(4, 4+self.get_face_pionts*2, 2):
                output[..., i:i+2] = (output[..., i:i+2] + grid) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        if self.get_face_pionts:#outputs: [xc,yc,w,h, [x,y]*6 ,obj, cls...]
            for i in range(4, 4+self.get_face_pionts*2, 2):
                outputs[..., i:i+2] = (outputs[..., i:i+2] + grids) * strides
            # org_outputs = torch.cat([outputs[:, :, :4], outputs[:, :, -self.num_classes-1:]], 2)

        return outputs
        # return org_outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        '''
        '''
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, -self.num_classes-1].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, -self.num_classes:]  # [batch, n_anchors_all, n_cls]
        if self.get_face_pionts: #outputs: [xc,yc,w,h, [x,y]*6 ,obj, cls...]
            points_preds = outputs[:, :, 4:4+2*self.get_face_pionts]
        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        points_targets = []
        fg_masks = []
        fg_iou_metrics = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                if self.get_face_pionts:
                    points_target = outputs.new_zeros((0, 3*self.get_face_pionts)) #[x,y,score]*6
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                fg_iou_metric = outputs.new_zeros((0))
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                if self.get_face_pionts: #labels: [c,xc,yc,w,h, [x,y,score]*6]
                    points_per_image = labels[batch_idx, :num_gt, 5:5+self.get_face_pionts*3]

                try:
                    '''SimOTA or Task Alignment Learning(TAL)'''
                    (
                        gt_matched_classes,
                        fg_mask,
                        fg_iou_metric,
                        matched_gt_inds,
                        num_fg_img,
                        all_targets,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )

                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        fg_iou_metric,
                        matched_gt_inds,
                        num_fg_img,
                        all_targets,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target, obj_target, reg_target = all_targets
                # cls_target = F.one_hot(
                #     gt_matched_classes.to(torch.int64), self.num_classes
                # ) * pred_ious_this_matching.unsqueeze(-1)  # 说明YOLOX的cls_score带有iou感知功能
                # obj_target = fg_mask.unsqueeze(-1)
                # reg_target = gt_bboxes_per_image[matched_gt_inds]  # 该特征层匹配到的gt索引
                if self.get_face_pionts:
                    points_target = points_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            try:
                fg_iou_metrics.append(fg_iou_metric)
            except:
                print('error')
            if self.use_l1:
                l1_targets.append(l1_target)
            if self.get_face_pionts:
                points_targets.append(points_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        fg_iou_metrics = torch.cat(fg_iou_metrics, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        if self.get_face_pionts:
            points_targets = torch.cat(points_targets, 0)
        num_fg = max(num_fg, 1)
        # if self.Assigner != 'TAL':
        #     fg_iou_metrics = torch.ones_like(fg_iou_metrics)
        loss_iou = (self.iou_loss_fn(bbox_preds.view(-1, 4)[fg_masks], reg_targets))\
                       .sum() / num_fg

        if self.reg_iou:
            loss_obj = (self.obj_loss_fn(obj_preds.view(-1, 1), obj_targets))\
                       .sum() / num_fg
            loss_cls = (self.cls_loss_fn(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets))\
                       .sum() / num_fg
        else:
            loss_obj = 0.
            # loss_cls_old = (torch.nn.functional.binary_cross_entropy_with_logits(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets, reduction='none')) \
            #                .sum() / num_fg
            loss_cls = (self.cls_loss_fn(cls_preds.view(-1, self.num_classes), cls_targets, fg_masks))\
                       .sum() / num_fg
            # print(f'cls loss: {loss_cls_old:.4f} {loss_cls:.4f}')
        if self.use_l1:
            loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets))\
                          .sum() / num_fg
        else:
            loss_l1 = 0.0

        if self.get_face_pionts:
            loss_points = (self.points_loss_fn(points_preds.view(-1, 2*self.get_face_pionts)[fg_masks], points_targets))\
                              .sum() / num_fg
        else:
            loss_points = 0.0

        loss = self.box_loss_weight * loss_iou + loss_obj + self.cls_loss_weight*loss_cls + loss_l1 + loss_points*self.points_loss_weight

        return (
            loss,
            self.box_loss_weight * loss_iou,
            loss_obj,
            loss_cls*self.cls_loss_weight,
            loss_l1,
            loss_points*self.points_loss_weight,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def SimOTA(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            if self.reg_iou:
                cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                )
            else:
                cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        cls_target = F.one_hot(
            gt_matched_classes.to(torch.int64), self.num_classes
        ) * pred_ious_this_matching.unsqueeze(-1)  # 说明YOLOX的cls_score带有iou感知功能
        obj_target = fg_mask.unsqueeze(-1)
        reg_target = gt_bboxes_per_image[matched_gt_inds]  # 该特征层匹配到的gt索引

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
            (cls_target, obj_target, reg_target)
        )
    @torch.no_grad()
    def TaskAlignedAssigner(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):
        """
        对每张图使用TOOD中的TAL进行匹配，代码参考YOLOv6
        """
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        """各个函数模块对应YOLO v6中的TAL"""
        ''' get_pos_mask 函数-------------------------------------------'''
        # get anchor_align metric
        overlaps = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        if self.reg_iou:
            cls_preds_score = cls_preds[batch_idx].sigmoid()*obj_preds[batch_idx].sigmoid()
        else:
            cls_preds_score = cls_preds[batch_idx].sigmoid()
        bbox_scores = cls_preds_score[:, gt_classes.to(torch.long)]  ################# 该步存疑，未做验证
        bbox_scores = bbox_scores.permute(1, 0)
        align_metric = bbox_scores.pow(self.TAL_alpha) * overlaps.pow(self.TAL_beta)

        # get in_gts mask 锚点在gt框内就算
        mask_in_gts = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
            only_in_gt=True  # TAL 该步要求锚点在GT内部即可，无中心要求
        )  #### 函数出现过的特殊情况：可能会出现GT框内无锚点，这是因为gt框太细长，夹在锚点们之间

        # get topk_metric mask
        topk_metrics, topk_idxs = torch.topk(align_metric * mask_in_gts, self.TAL_topk, axis=-1, largest=True)
        anchor_points_is_topk = torch.zeros([num_gt, total_num_anchors]).to(mask_in_gts.device)
        for i in range(num_gt):
            anchor_points_is_topk[i, topk_idxs[i]] = 1.
        mask_pos = anchor_points_is_topk * mask_in_gts
        '''get_pos_mask 函数输出 mask_pos, align_metric, overlaps----------'''

        '''select_highest_overlaps 函数-----------------------------------'''
        """if an anchor box is assigned to multiple gts, the one with the highest iou will be selected."""
        fg_mask = mask_pos.sum(axis=-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(0) > 1).repeat([num_gt, 1])
            max_overlaps_idx = overlaps.argmax(axis=0)
            is_max_overlaps = F.one_hot(max_overlaps_idx, num_gt).to(overlaps.dtype)
            is_max_overlaps = is_max_overlaps.permute(1, 0)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask = mask_pos.sum(axis=-2)
        target_gt_idx = mask_pos.argmax(axis=-2)
        '''select_highest_overlaps函数输出 target_gt_idx, fg_mask, mask_pos'''

        ''' get_targets函数--------------------------------------------------'''
        # assigned target labels
        target_labels = gt_classes.long().flatten()[target_gt_idx]
        # assigned target boxes
        target_bboxes = gt_bboxes_per_image.reshape([-1, 4])[target_gt_idx]
        # assigned target scores
        target_scores = F.one_hot(target_labels, self.num_classes)
        fg_scores_mask = fg_mask[:, None].repeat(1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores,
                                        torch.full_like(target_scores, 0))
        '''get_targets函数输出target_labels, target_bboxes, target_scores-----'''

        '''normalize'''
        align_metric *= mask_pos
        pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.TAL_eps)).max(-2)[0].unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        ''''''
        fg_mask = fg_mask.bool()
        return (
            target_labels[fg_mask],  # gt_matched_classes,
            fg_mask,
            norm_align_metric[fg_mask],
            target_gt_idx[fg_mask],  # matched_gt_inds,
            fg_mask.sum(),
            (target_scores[fg_mask], fg_mask.unsqueeze(-1), target_bboxes[fg_mask]),
        )


    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
        only_in_gt=False,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        if only_in_gt:
            return is_in_boxes.to(bbox_deltas.dtype)
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds