##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
import math


class RFConv2d(Conv2d):
    """Rectified Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 average_mode=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.rectify = average_mode or (padding[0] > 0 or padding[1] > 0)
        self.average = average_mode

        super(RFConv2d, self).__init__(
                 in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups,
                 bias=bias, padding_mode=padding_mode)

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        output = self._conv_forward(input, self.weight)
        if self.rectify:
            output = rectify(output, input, self.kernel_size, self.stride,
                             self.padding, self.dilation, self.average)
        return output

    def extra_repr(self):
        return super().extra_repr() + ', rectify={}, average_mode={}'. \
            format(self.rectify, self.average)


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, channel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, int(channel//self.radix), dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        if self.radix > 1:
            atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        else:
            atten = F.sigmoid(atten, dim=1).view(batch, -1, 1, 1)

        if self.radix > 1:
            atten = torch.split(atten, int(channel//self.radix), dim=1)
            out = sum([att*split for (att, split) in zip(atten, splited)])
        else:
            out = atten * x
        return out.contiguous()


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

# https://github.com/zhanghang1989/PyTorch-Encoding/blob/b872eb8cc01f6e0dde3d9acfb5846abe5c3a450a/encoding/nn/dropblock.py
# class DropBlock2D(nn.Module):
#     r"""Randomly zeroes 2D spatial blocks of the input tensor.
#     As described in the paper
#     `DropBlock: A regularization method for convolutional networks`_ ,
#     dropping whole blocks of feature map allows to remove semantic
#     information as compared to regular dropout.
#     Args:
#         drop_prob (float): probability of an element to be dropped.
#         block_size (int): size of the block to drop
#     Shape:
#         - Input: `(N, C, H, W)`
#         - Output: `(N, C, H, W)`
#     .. _DropBlock: A regularization method for convolutional networks:
#        https://arxiv.org/abs/1810.12890
#     """

#     def __init__(self, drop_prob, block_size, share_channel=False):
#         super(DropBlock2D, self).__init__()
#         self.register_buffer('i', torch.zeros(1, dtype=torch.int64))
#         self.register_buffer('drop_prob', drop_prob * torch.ones(1, dtype=torch.float32))
#         self.inited = False
#         self.step_size = 0.0
#         self.start_step = 0
#         self.nr_steps = 0
#         self.block_size = block_size
#         self.share_channel = share_channel

#     def reset(self):
#         """stop DropBlock"""
#         self.inited = True
#         self.i[0] = 0
#         self.drop_prob = 0.0

#     def reset_steps(self, start_step, nr_steps, start_value=0, stop_value=None):
#         self.inited = True
#         stop_value = self.drop_prob.item() if stop_value is None else stop_value
#         self.i[0] = 0
#         self.drop_prob[0] = start_value
#         self.step_size = (stop_value - start_value) / nr_steps
#         self.nr_steps = nr_steps
#         self.start_step = start_step

#     def forward(self, x):
#         if not self.training or self.drop_prob.item() == 0.:
#             return x
#         else:
#             self.step()

#             # get gamma value
#             gamma = self._compute_gamma(x)

#             # sample mask and place on input device
#             if self.share_channel:
#                 mask = (torch.rand(x.shape[0], *x.shape[2:], device=x.device, dtype=x.dtype) < gamma).squeeze(1)
#             else:
#                 mask = (torch.rand(*x.shape, device=x.device, dtype=x.dtype) < gamma)

#             # compute block mask
#             block_mask, keeped = self._compute_block_mask(mask)

#             # apply block mask
#             out = x * block_mask

#             # scale output
#             out = out * (block_mask.numel() / keeped).to(out)
#             return out

#     def _compute_block_mask(self, mask):
#         block_mask = F.max_pool2d(mask,
#                                   kernel_size=(self.block_size, self.block_size),
#                                   stride=(1, 1),
#                                   padding=self.block_size // 2)

#         keeped = block_mask.numel() - block_mask.sum().to(torch.float32)
#         block_mask = 1 - block_mask

#         return block_mask, keeped

#     def _compute_gamma(self, x):
#         _, c, h, w = x.size()
#         gamma = self.drop_prob.item() / (self.block_size ** 2) * (h * w) / \
#             ((w - self.block_size + 1) * (h - self.block_size + 1))
#         return gamma

#     def step(self):
#         assert self.inited
#         idx = self.i.item()
#         if idx > self.start_step and idx < self.start_step + self.nr_steps:
#             self.drop_prob += self.step_size
#         self.i += 1

#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         idx_key = prefix + 'i'
#         drop_prob_key = prefix + 'drop_prob'
#         if idx_key not in state_dict:
#             state_dict[idx_key] =  torch.zeros(1, dtype=torch.int64)
#         if idx_key not in drop_prob_key:
#             state_dict[drop_prob_key] =  torch.ones(1, dtype=torch.float32)
#         super(DropBlock2D, self)._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict,
#             missing_keys, unexpected_keys, error_msgs)

#     def _save_to_state_dict(self, destination, prefix, keep_vars):
#         """overwrite save method"""
#         pass

#     def extra_repr(self):
#         return 'drop_prob={}, step_size={}'.format(self.drop_prob, self.step_size)

# def reset_dropblock(start_step, nr_steps, start_value, stop_value, m):
#     """
#     Example:
#         from functools import partial
#         apply_drop_prob = partial(reset_dropblock, 0, epochs*iters_per_epoch, 0.0, 0.1)
#         net.apply(apply_drop_prob)
#     """
#     if isinstance(m, DropBlock2D):
#         print('reseting dropblock')
#         m.reset_steps(start_step, nr_steps, start_value, stop_value)

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix > 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            self.conv2 = RFConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 1:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNest(nn.Module):
    """ResNet Variants
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNest, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        self.num_classes = num_classes
        if self.num_classes > 0:
            self.avgpool = GlobalAvgPool2d()
            self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
            self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.num_classes > 0:
            x = self.layer1(x)  # 56, 4x
            x = self.layer2(x)  # 28, 8x
            x = self.layer3(x)  # 14, 16x
            x = self.layer4(x)  # 7, 32x

            x = self.avgpool(x)  # 1
            #x = x.view(x.size(0), -1)
            x = torch.flatten(x, 1)
            if self.drop:
                x = self.drop(x)
            x = self.fc(x)

            return x
        else:
            layers = []

            x = self.layer1(x)
            layers.append(x)
            x = self.layer2(x)
            layers.append(x)
            x = self.layer3(x)
            layers.append(x)
            x = self.layer4(x)
            layers.append(x)

            return layers


__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

_url_format = 'https://hangzh.s3.amazonaws.com/encoding/models/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNest(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            # resnest_model_urls['resnest50'], progress=True, check_hash=True))
            resnest_model_urls['resnest50'], progress=True))
    return model

def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNest(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            # resnest_model_urls['resnest101'], progress=True, check_hash=True))
            resnest_model_urls['resnest101'], progress=True))
    return model

def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNest(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))
    return model

def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNest(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest269'], progress=True, check_hash=True))
    return model


if __name__=='__main__':
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter("/home/nie/f/test") as writer:
        x = torch.randn((4,3,224,224))
        model = resnest50(pretrained=False)
        # model.eval()
        y = model(x)
        writer.add_graph(model, x)