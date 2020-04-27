# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from fcos_core.modeling import registry
from fcos_core.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import hrfpn as hrfpn_module
from . import resnet
from . import hrnet
from . import mobilenet
from . import fbnet
from . import dla_lite
from . import efficientnet_model
from . import resnest


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-22-FPN-RETINANET")
@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-ST-50-FPN-RETINANET")
def build_resnest_fpn_p3p7_backbone(cfg):
    body = resnest.resnest50(num_classes=-1)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("MNV2-FPN-RETINANET")
def build_mnv2_fpn_backbone(cfg):
    body = mobilenet.MobileNetV2(cfg)
    in_channels_stage2 = body.return_features_num_channels
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2[1],
            in_channels_stage2[2],
            in_channels_stage2[3],
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(out_channels, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("HRNET-W18")
@registry.BACKBONES.register("HRNET-W32")
@registry.BACKBONES.register("HRNET-W40")
def build_hrnet_fpn_backbone(cfg):
    if cfg.MODEL.BACKBONE.CONV_BODY == "HRNET-W18":
        width = 18
    elif cfg.MODEL.BACKBONE.CONV_BODY == "HRNET-W32":
        width = 32
    elif cfg.MODEL.BACKBONE.CONV_BODY == "HRNET-W40":
        width = 40
    else:
        raise NotImplementedError

    hrnet_args = dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(4,),
            num_channels=(64,),
            fuse_method='SUM'),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='BASIC',
            num_blocks=(4, 4),
            num_channels=(width, width * 2),
            fuse_method='SUM'),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(width, width * 2, width * 4),
            fuse_method='SUM'),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(width, width * 2, width * 4, width * 8),
            fuse_method='SUM')
    )
    fpn_in_channels = [width, width * 2, width * 4, width * 8]

    hrnet.BatchNorm2d = nn.SyncBatchNorm if cfg.MODEL.SYNCBN else nn.BatchNorm2d
    body = hrnet.HighResolutionNet(extra=hrnet_args)
    fpn = getattr(hrfpn_module, cfg.MODEL.HRNET.FPN.TYPE)(
        in_channels=fpn_in_channels, 
        out_channels=cfg.MODEL.HRNET.FPN.OUT_CHANNEL,
        conv_stride=cfg.MODEL.HRNET.FPN.CONV_STRIDE,
        num_level=len(cfg.MODEL.FCOS.FPN_STRIDES),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = cfg.MODEL.HRNET.FPN.OUT_CHANNEL
    return model


@registry.BACKBONES.register("FBNet-FPN-RETINANET")
def build_fbnet_fpn_p3p7_backbone(cfg):
    body = fbnet.add_conv_body_fpn(cfg)
    out_channels = 64
    in_channels_p6p7 = 64
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            64,
            88,
            104,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("DLA34-FPN-RETINANET")
def build_dla34_fpn_p3p7_backbone(cfg):
    body = dla_lite.dla34_fpn()
    out_channels = 128
    in_channels_p6p7 = 128
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            128,
            256,
            512,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("EFFICIENTNET-B3-FPN-RETINANET")
def build_efficientnet_b3_fpn_p3p7_backbone(cfg):
    body = efficientnet_model.EfficientNet.efficientnet_b3_fpn()
    out_channels = 128
    in_channels_p6p7 = 128
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            48,
            136,
            384,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("EFFICIENTNET-B3-HRFPN-RETINANET")
def build_efficientnet_b3_hrfpn_p3p7_backbone(cfg):
    body = efficientnet_model.EfficientNet.efficientnet_b3_fpn()
    out_channels = 128
    fpn = fpn_module.HRFPN_Block(
        in_channels_list=[
            0,
            48,
            136,
            384,
        ],
        out_channels=out_channels,
        num_outs=5,
        pooling_type='AVG'
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model



def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
