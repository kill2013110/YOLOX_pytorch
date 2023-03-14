#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
try:
    from .darknet import CSPDarknet
    from .network_blocks import BaseConv, CSPLayer, DWConv
    from .convnextv2 import convnextv2_atto, convnextv2_femto
except:
    from darknet import CSPDarknet
    from network_blocks import BaseConv, CSPLayer, DWConv
    from convnextv2 import convnextv2_atto, convnextv2_femto

class ConvNextv2_PAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        spp_size=(5, 9, 13),
        backbone_ckpt=None,
        convnextv2='atto'
    ):
        super().__init__()
        if convnextv2=='atto':
            self.backbone = convnextv2_atto(classifier=False)
            self.backbone_channel = [80, 160, 320]
        else:
            pass
        if backbone_ckpt!=None:
            self.backbone.load_state_dict(torch.load(backbone_ckpt)['model'], False)
            print('convnextv2 atto backbone_ckpt loaded !')

        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        '''convnextv2 backbone channel adjust'''
        self.C_s = CSPLayer(
            int(self.backbone_channel[2]), int(in_channels[2] * width),
            round(3 * depth), False, depthwise=depthwise, act=act,)  # cat
        self.C_m = CSPLayer(
            int(self.backbone_channel[1]), int(in_channels[1] * width),
            round(3 * depth), False, depthwise=depthwise, act=act,)  # cat
        self.C_l = CSPLayer(
            int(self.backbone_channel[0]), int(in_channels[0] * width),
            round(3 * depth), False, depthwise=depthwise, act=act,)  # cat

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )  # x1
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
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        # features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = out_features
        [x2, x1, x0] = self.C_l(x2), self.C_m(x1), self.C_s(x0)

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8  # fpn_out2

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
if __name__ == "__main__":
    import torch, copy
    # from torchinfo import summary
    from thop import profile
    net = ConvNextv2_PAFPN(depth=0.33, width=0.5)
    a = net(torch.randn([4, 3, 640, 640]))

    for i in a:
        print(i.shape)

    img = torch.zeros((4, 3, 512, 640), device=next(net.parameters()).device)
    flops, params = profile(copy.deepcopy(net), inputs=(img,), verbose=False)
    print(f' flops:{flops / 1e9} f')
    print(f' params: {params / 1e6} M')