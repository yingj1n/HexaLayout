### This file is referenced from `https://arxiv.org/pdf/2002.08394.pdf`
### The github repo can be found here: https://github.com/hbutsuak95/monolayout

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

import numpy as np
from collections import OrderedDict

from module_unet import UNet
from module_resnet import ResNetEncoder
import utils


##################### Building Blocks #######################


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out



def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

##################### ORG LANE LANE DETECTION MODEL #############
class SingleImageCNN(nn.Module):
    def __init__(self, blocks_sizes=[64, 128, 256], depths=[2, 2, 2]):
        super(SingleImageCNN, self).__init__()
        self.convencoder = ResNetEncoder(3, blocks_sizes=blocks_sizes, depths=depths)

    def forward(self, x, verbose=False):
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.convencoder(x)
        if verbose:
            print('single_out', x.shape)
        return x

class UNetRoadMapNetwork(nn.Module):
    def __init__(self,
                 single_blocks_sizes=[16, 32, 64],
                 single_depths=[2, 2, 2],
                 unet_start_filts=64,
                 unet_depth=5):
        super(UNetRoadMapNetwork, self).__init__()

        self.single_encoder = SingleImageCNN(
            blocks_sizes=single_blocks_sizes,
            depths=single_depths)

        self.u_net = UNet(num_classes=1,
                          in_channels=single_blocks_sizes[-1],
                          depth=unet_depth,
                          start_filts=unet_start_filts, up_mode='transpose',
                          merge_mode='concat')

    def forward(self, single_cam_input, verbose=False):
        encoder_outputs = []
        for idx, cam_input in enumerate(single_cam_input):
            output = self.single_encoder(cam_input, verbose)
            encoder_outputs.append(output)

        x = utils.combine_six_to_one(encoder_outputs)

        if verbose:
            print('concat_single', x.shape)

        x = self.u_net(x, verbose)

        if verbose:
            print('unet_output', x.shape)

        x = F.interpolate(x, size=(800, 800), mode='bilinear', align_corners=False)

        if verbose:
            print('interpolate', x.shape)

        return x



##################### ENCODER ####################


'''
ORG: 
A context encoder: extracts multi-scale feature representations from the input monocular image. 
This provides a shared context that captures static as well as dynamic scene components for subsequent processing.
CHANGE: 
- process multiple images 
- static and dyanamic are captured in one scence 
'''

# class ResNetMultiImageInput(models.ResNet):
#     """Constructs a resnet model with varying number of input images.
#     Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#     """
#     def __init__(self, block, layers, num_classes=1000, num_input_images=1):
#         super(ResNetMultiImageInput, self).__init__(block, layers)
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(
#             num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


# def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
#     """Constructs a ResNet model.
#     Args:
#         num_layers (int): Number of resnet layers. Must be 18 or 50
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         num_input_images (int): Number of frames stacked as input
#     """
#     assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
#     blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
#     block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
#     model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

#     ### Change: pretrain should always be false 
#     # if pretrained:
#     #     loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
#     #     loaded['conv1.weight'] = torch.cat(
#     #         [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
#     #     model.load_state_dict(loaded)
#     return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features




############ Decoder & Discriminator ################
'''
Two decoders: 
1)  An amodal static scene decoder which decodes the shared context to produce an amodal layout of the static scene. 
This model consists of a series of deconvolution and upsampling layers that map the shared context to a static scene 
bird’s eye view.
2). A dynamic scene decoder which is architecturally similar to the road decoder and predicts the vehicle occupancies 
in bird’s eye view.

Ignore discriminator for now 
'''


class Decoder(nn.Module):
    def __init__(self, num_ch_enc):
        super(Decoder, self).__init__()
        self.num_output_channels = 2
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_ch_concat = np.array([64, 128, 256, 512, 128])
        self.conv_mu = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv_log_sigma = nn.Conv2d(128, 128, 3, 1, 1)
        outputs = {}
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = 128 if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            num_ch_concat = self.num_ch_concat[i]
            self.convs[("upconv", i, 0)] = nn.Conv2d(num_ch_in, num_ch_out, 3, 1, 1) #Conv3x3(num_ch_in, num_ch_out)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] =  nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(num_ch_out, num_ch_out, 3, 1, 1) #ConvBlock(num_ch_out, num_ch_out)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)

        self.convs["topview"] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        self.dropout = nn.Dropout3d(0.2)
        self.decoder = nn.ModuleList(list(self.convs.values()))



    def forward(self, x, is_training=True):
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)
            x = upsample(x)
            #x = torch.cat((x, features[i-6]), 1)
            x = self.convs[("upconv", i, 1)](x)
            x = self.convs[("norm", i, 1)](x)

        if is_training:
                x = self.convs["topview"](x) #self.softmax(self.convs["topview"](x))
        else:
                softmax = nn.Softmax2d()
                x = softmax(self.convs["topview"](x))
        #outputs["car"] = x
        return x #outputs


class Encoder(nn.Module):
    def __init__(self, num_layers, img_ht, img_wt, pretrained=True):
        super(Encoder, self).__init__()

        self.resnet_encoder = ResnetEncoder(num_layers, pretrained)#opt.weights_init == "pretrained"))
        num_ch_enc = self.resnet_encoder.num_ch_enc
        #convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(num_ch_enc[-1], 128)
        self.conv2 = Conv3x3(128, 128)
        self.pool = nn.MaxPool2d(2)

        #fully connected
        curr_h = img_ht//(2**6)
        curr_w = img_wt//(2**6)
        features_in = curr_h*curr_w*128
        self.fc_mu = torch.nn.Linear(features_in, 2048)
        self.fc_sigma = torch.nn.Linear(features_in, 2048)
        self.fc = torch.nn.Linear(features_in, 2048)
        

    def forward(self, x, is_training= True):

        batch_size, c, h, w = x.shape
        x = self.resnet_encoder(x)[-1]
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = self.pool(x)
        return x
