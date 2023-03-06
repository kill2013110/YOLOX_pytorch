#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
try:
    from .darknet import CSPDarknet, CSPDarknet_4
    from .network_blocks import BaseConv, CSPLayer, DWConv
except:
    from darknet import CSPDarknet, CSPDarknet_4
    from network_blocks import BaseConv, CSPLayer, DWConv

class YOLO_fpn_TSCODE(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark2", "dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        spp_size=(5, 9, 13),
    ):
        super().__init__()
        self.backbone = CSPDarknet_4(depth, width, depthwise=depthwise, act=act, spp_size=spp_size)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            # int(2 * in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            # int(2 * in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        '''类似于RetinaNet的最小特征图'''
        self.smallest_conv = BaseConv(
            int(in_channels[2] * width), int(in_channels[2] * width)*2, ksize=3, stride=2, act=act
        )
        # # bottom-up conv
        # self.bu_conv2 = Conv(
        #     int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        # )
        # self.C3_n3 = CSPLayer(
        #     int(2 * in_channels[0] * width),
        #     int(in_channels[1] * width),
        #     round(3 * depth),
        #     False,
        #     depthwise=depthwise,
        #     act=act,
        # )
        #
        # # bottom-up conv
        # self.bu_conv1 = Conv(
        #     int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        # )
        # self.C3_n4 = CSPLayer(
        #     int(2 * in_channels[1] * width),
        #     int(in_channels[2] * width),
        #     round(3 * depth),
        #     False,
        #     depthwise=depthwise,
        #     act=act,
        # )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x3, x2, x1, x0] = features  # 与TSCODE的图相反，号约大特征图尺寸越大

        x00 = self.smallest_conv(x0)

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8 # fpn_out2

        # p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        # p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        # pan_out1 = self.C3_n3(p_out1)  # 512->512/16
        #
        # p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        # pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (x00, x0, f_out0, pan_out2, x3)  # 特征图由小到大排列
        return outputs

if __name__ == "__main__":
    net = YOLO_fpn_TSCODE(depth=0.33, width=0.25)
    a = net(torch.randn([4, 3, 416, 416]))
    print('123')