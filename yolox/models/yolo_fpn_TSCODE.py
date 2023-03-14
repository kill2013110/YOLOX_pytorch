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

        '''''''基于pan进行裁剪和修改'''''''
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        # 多了个降维模块
        self.down_c = CSPLayer(
            int(in_channels[1] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # 多了个最大特征图升维模块
        self.up_c = CSPLayer(
            int(in_channels[0] * width * 0.5),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # 多了个生成最小特征图的卷积
        self.smallest_outconv = BaseConv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )

        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
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
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
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
        '''  基于pan进行裁剪 '''
        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x3, x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        # 通道数256
        fpn_out0_slim = self.down_c(fpn_out0)
        # 通道数整合
        x00 = self.smallest_outconv(fpn_out0_slim)

        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        # 通道数128

        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8  # fpn_out2
        # 通道数128

        # p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        # 通道数128

        # p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        # pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        # p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        # pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        x3 = self.up_c(x3)
        outputs = (x3, pan_out2, fpn_out1, fpn_out0_slim, x00)
        return outputs
if __name__ == "__main__":
    import torch, copy
    # from torchinfo import summary
    from thop import profile
    net = YOLO_fpn_TSCODE(depth=0.33, width=0.5)
    a = net(torch.randn([4, 3, 640, 640]))

    for i in a:
        print(i.shape)

    img = torch.zeros((4, 3, 640, 640), device=next(net.parameters()).device)
    flops, params = profile(copy.deepcopy(net), inputs=(img,), verbose=False)
    print(flops)
    print(f'{params/1e6} M')
    # print('123')