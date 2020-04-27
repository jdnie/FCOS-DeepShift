# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            # inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            inner_top_down = F.interpolate(
                last_inner, size=(int(inner_lateral.shape[-2]), int(inner_lateral.shape[-1])),
                mode='nearest'
            )
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class HRFPN_Block(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_outs=5, pooling_type='AVG'):
        super(HRFPN_Block, self).__init__()

        self.in_channels_list = in_channels_list
        self.num_outs = num_outs

        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(in_channels_list),
                      out_channels=out_channels,
                      kernel_size=1),
        )

        self.fpn_conv = nn.ModuleList()
        for i in range(num_outs):
            self.fpn_conv.append(nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ))
        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        if inputs.__len__() >= self.in_channels_list.__len__():
            inputs = inputs[inputs.__len__() - self.in_channels_list.__len__():]

        outs = []
        out_start = False
        start_idx = 0
        for i in range(self.in_channels_list.__len__()):
            if self.in_channels_list[i] == 0:
                continue
            if not out_start:
                outs.append(inputs[i])
                out_start = True
                start_idx = i
            else:
                outs.append(F.interpolate(inputs[i], scale_factor=2**(i-start_idx),
                                          mode='bilinear', align_corners=True))
        out = torch.cat(outs, dim=1)
        out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_outs):
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        outputs = []
        for i in range(self.num_outs):
            outputs.append(self.fpn_conv[i](outs[i]))
        return tuple(outputs)

