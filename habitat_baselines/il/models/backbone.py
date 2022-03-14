# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""


from habitat import logger

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
# from timm.models import create_model
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet101

from .misc import NestedTensor

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning to make it user-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2",
                             "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone,
                                            return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        out = OrderedDict()
        for name, x in xs.items():
            mask = F.interpolate(
                tensor_list.mask[None].float(), size=x.shape[-2:]
            ).bool()[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool,
                 return_interm_layers: bool, dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels,
                         return_interm_layers)


class GroupNorm32(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, **kargs):
        super().__init__(num_groups, num_channels, **kargs)


class GroupNormBackbone(BackboneBase):
    """ResNet backbone with GroupNorm with 32 channels."""

    def __init__(self, name: str, train_backbone: bool,
                 return_interm_layers: bool, dilation: bool):
        name_map = {
            "resnet50-gn": (
                "resnet50",
                 "/checkpoint/szagoruyko/imagenet/22014122/checkpoint.pth"
            ),
            "resnet101-gn": (
                "resnet101",
                "/checkpoint/szagoruyko/imagenet/22080524/checkpoint.pth"
            ),
        }
        backbone = getattr(torchvision.models, name_map[name][0])(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=GroupNorm32
        )
        checkpoint = torch.load(name_map[name][1], map_location="cpu")
        state_dict = {k[7:]: p for k, p in checkpoint["model"].items()}
        backbone.load_state_dict(state_dict)
        num_channels = \
            512 if name_map[name][0] in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels,
                         return_interm_layers)


def replace_bn(m, name=""):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if isinstance(target_attr, torch.nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(target_attr.num_features)
            bn = getattr(m, attr_str)
            frozen.weight.data.copy_(bn.weight)
            frozen.bias.data.copy_(bn.bias)
            frozen.running_mean.data.copy_(bn.running_mean)
            frozen.running_var.data.copy_(bn.running_var)
            setattr(m, attr_str, frozen)
    for n, ch in m.named_children():
        replace_bn(ch, n)


class GN_8(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.gn = torch.nn.GroupNorm(8, num_channels)

    def forward(self, x):
        return self.gn(x)


# class TimmBackbone(nn.Module):
#     def __init__(self, name, return_interm_layers, main_layer=-1, group_norm=False):
#         super().__init__()
#         backbone = create_model(name, pretrained=True, in_chans=3, features_only=True, out_indices=(1, 2, 3, 4))
#
#         with torch.no_grad():
#             replace_bn(backbone)
#         num_channels = backbone.feature_info.channels()[-1]
#         self.body = backbone
#         self.num_channels =  num_channels
#         self.interm = return_interm_layers
#         self.main_layer = main_layer
#
#     def forward(self, tensor_list):
#         xs = self.body(tensor_list.tensors)
#         if not self.interm:
#             xs = [xs[self.main_layer]]
#         out = OrderedDict()
#         for i, x in enumerate(xs):
#             mask = F.interpolate(tensor_list.mask[None].float(), size=x.shape[-2:]).bool()[0]
#             out[f"layer{i}"] = NestedTensor(x, mask)
#         return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        xs = self[0](tensor_list)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


# TODO: how do I correctly print the following?
# Copyright (c) Svyatoslav Pchelintsev. Licensed under the Apache License 2.0.
# All Rights Reserved.


class UNETEncoderBlock(nn.Module):
    """
    The building brick of the left part of the UNET architecture.
    Batch normalization is added as it improves performance of the unit.
    """

    def __init__(self, in_channels, out_channels):
        super(UNETEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))


class UNETDecoderBlock(nn.Module):
    """
    Building brick of the right part of the UNET architecture.
    """

    def __init__(self, in_channels, out_channels, inter_channels=None):
        super(UNETDecoderBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )

        if inter_channels is None:
            inter_channels = 2 * out_channels
        self.conv_block = UNETEncoderBlock(inter_channels, out_channels)

    def forward(self, x, skip_conn):
        x = self.up_conv(x)
        return self.conv_block(torch.cat([skip_conn, x], dim=1))


class UNETDecoder(nn.Module):
    """
    The right part of the UNET architecture. Returns reconstruction for
    semantic segmentation and rgb, depth image.
    """

    def __init__(self, seg_channels, depth_channels=1, rgb_channels=3):
        super(UNETDecoder, self).__init__()

        # In case of input_image.shape = (3, 256, 256) we get
        # (16, 16) -> (32, 32):
        self.decoder_block1 = UNETDecoderBlock(1024, 256)
        # (32, 32) -> (64, 64):
        self.decoder_block2 = UNETDecoderBlock(256, 128)
        # (64, 64) -> (128, 128):
        self.decoder_block3 = UNETDecoderBlock(128, 64)
        # (128, 128) -> (256, 256):
        self.decoder_block4 = UNETDecoderBlock(64, 32, inter_channels=35)

        self.final_conv_seg = nn.Conv2d(32, seg_channels, kernel_size=1)
        self.final_conv_depth = nn.Conv2d(32, depth_channels, kernel_size=1)
        self.final_conv_rgb = nn.Conv2d(32, rgb_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skip_conns):
        x = self.decoder_block1(x, skip_conns[-1])
        x = self.decoder_block2(x, skip_conns[-2])
        x = self.decoder_block3(x, skip_conns[-3])
        x = self.decoder_block4(x, skip_conns[-4])

        output_seg = self.sigmoid(self.final_conv_seg(x))
        output_depth = self.sigmoid(self.final_conv_depth(x))
        output_rgb = self.sigmoid(self.final_conv_rgb(x))
        return output_seg, output_depth, output_rgb


class MultitaskResNet101(nn.Module):
    """
    CNN module. UNET architecture that utilizes parts of ResNet101 as encoder
    and bridge (bottleneck). Numbers in names mean ordinal numbers from the
    output of torchsummary.summary(resnet101, (3, 256, 256)).
    Note that torchvision.models.resnet101
    is loaded always pretrained as we don't have an intention to train it from
    nothing.
    """

    def __init__(
        self,
        num_classes: int = 41,
        only_encoder: bool = False,
        pretrained: bool = False,
        checkpoint_path: str = None,
        freeze_encoder: bool = False
    ):
        super(MultitaskResNet101, self).__init__()

        self.num_classes = num_classes
        self.only_encoder = only_encoder

        self.resnet = resnet101(pretrained=True)
        self.decoder = UNETDecoder(seg_channels=self.num_classes)

        self.avg_pool = nn.AdaptiveMaxPool2d((4, 4))

        # TODO: check weights initialization
        if self.only_encoder:
            if pretrained:
                assert checkpoint_path is None, \
                    "If CNN is used as encoder, a checkpoint must be provided!"

                logger.info(
                    "Loading resnet-based CNN weights from {}".format(
                        checkpoint_path
                    )
                )
                checkpoint = torch.load(
                    checkpoint_path, map_location={"cuda:0": "cpu"}
                )
                self.load_state_dict(checkpoint["model"])

                if freeze_encoder:
                    for param in self.parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor):
        ############## ENCODER ##############
        feature_maps = [x, ]
        feat_map1 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        feature_maps.append(feat_map1)  # ReLU-3, before (128, 128) -> (64, 64)

        input_to_layer2 = self.resnet.layer1(self.resnet.maxpool(feat_map1))

        # Part of the zeroth Bottleneck of the SECOND layer
        conv2d_37 = self.resnet.layer2[0].conv1
        bn_38 = self.resnet.layer2[0].bn1
        relu_39 = self.resnet.layer2[0].relu
        feat_map2 = relu_39(bn_38(conv2d_37(input_to_layer2)))
        feature_maps.append(feat_map2)  # ReLU-39, before (64, 64) -> (32, 32)

        # Remained part of the zeroth Bottleneck
        conv2d_40 = self.resnet.layer2[0].conv2
        bn_41 = self.resnet.layer2[0].bn2
        relu_42 = self.resnet.layer2[0].relu
        conv2d_43 = self.resnet.layer2[0].conv3
        bn_44 = self.resnet.layer2[0].bn3
        conv2d_45_bn_46 = self.resnet.layer2[0].downsample
        relu_47 = self.resnet.layer2[0].relu
        out2 = bn_44(conv2d_43(relu_42(bn_41(conv2d_40(feat_map2)))))
        out2 += conv2d_45_bn_46(
            input_to_layer2  # Downsampling from the layer input
        )
        out2 = relu_47(out2)
        # Remained part of the second layer
        out2 = self.resnet.layer2[1](out2)
        out2 = self.resnet.layer2[2](out2)
        input_to_layer3 = self.resnet.layer2[3](out2)

        # Part of the zeroth Bottleneck of the THIRD layer
        conv2d_79 = self.resnet.layer3[0].conv1
        bn_80 = self.resnet.layer3[0].bn1
        relu_81 = self.resnet.layer3[0].relu
        feat_map3 = relu_81(bn_80(conv2d_79(input_to_layer3)))
        feature_maps.append(feat_map3)  # ReLU-81, before (32, 32) -> (16, 16)

        ############## BRIDGE ##############
        # Remained part of the zeroth Bottleneck
        conv2d_82 = self.resnet.layer3[0].conv2
        bn_83 = self.resnet.layer3[0].bn2
        relu_84 = self.resnet.layer3[0].relu
        conv2d_85 = self.resnet.layer3[0].conv3
        bn_86 = self.resnet.layer3[0].bn3
        conv2d_87_bn_88 = self.resnet.layer3[0].downsample
        relu_89 = self.resnet.layer3[0].relu
        out3 = bn_86(conv2d_85(relu_84(bn_83(conv2d_82(feat_map3)))))
        out3 += conv2d_87_bn_88(
            input_to_layer3  # Downsampling from the layer input
        )
        out3 = relu_89(out3)
        # Remained part of the third layer
        for i in range(1, len(self.resnet.layer3)):
            out3 = self.resnet.layer3[i](out3)

        # Currently only such architecture is supported
        assert len(feature_maps) == 4

        if self.only_encoder:
            return self.avg_pool(out3).reshape(-1, 1024 * 4 * 4)

        ############## DECODER ##############
        out_seg, out_depth, out_rgb = self.decoder(out3, feature_maps)
        return out_seg, out_depth, out_rgb


def build_backbone(args):
    if args.IL.CNN.cnn_model == "unet-resnet101":
        # TODO: do we need positional embedding here?
        # position_embedding = build_position_encoding(args)
        model_kwargs = {
            "only_encoder": True,
            "pretrained": True,
            "checkpoint_path": args.EQA_CNN_PRETRAIN_CKPT_PATH,
            "freeze_encoder": args.IL.CNN.freeze_encoder
        }
        model = MultitaskResNet101(**model_kwargs)
    elif "resnet" in args.IL.CNN.cnn_model:
        position_embedding = build_position_encoding(args.IL.CNN)
        train_backbone = not args.IL.CNN.freeze_encoder
        # TODO: do we need masks for segmentation?
        return_interm_layers = args.IL.CNN.masks
        # if args.backbone[: len("timm_")] == "timm_":
        #     backbone = TimmBackbone(
        #         args.backbone[len("timm_") :],
        #         return_interm_layers,
        #         main_layer=-1,
        #         group_norm=True,
        #     )
        if args.IL.CNN.cnn_model in ("resnet50-gn", "resnet101-gn"):
            backbone = GroupNormBackbone(
                args.IL.CNN.cnn_model, train_backbone, return_interm_layers,
                args.IL.CNN.dilation
            )
        else:
            backbone = Backbone(
                args.IL.CNN.cnn_model, train_backbone, return_interm_layers,
                args.IL.CNN.dilation
            )
        model = Joiner(backbone, position_embedding)
        model.num_channels = backbone.num_channels
    else:
        assert False, f"Unknown cnn_model name: {args.IL.CNN.cnn_model}"

    return model
