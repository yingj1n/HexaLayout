import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from collections import OrderedDict
from module_unet import UNet
#from module_resnet import ResNetEncoder
from resnet_encoder import ResnetEncoder

# Add decoders


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


class FusionNetwork(nn.Module):
    def __init__(self, in_feature=256, blocks_sizes=[256, 512], depths=[2, 2], out_features=512):
        super(FusionNetwork, self).__init__()
        self.n_feature = blocks_sizes[-1]

        self.convencoder = ResNetEncoder(in_feature, blocks_sizes=blocks_sizes, depths=depths)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # n_features, 1, 1

        self.linear = nn.Sequential(
            nn.Conv2d(self.n_feature, out_features, kernel_size=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(True),
        )

    #         self.fc = nn.Sequential(
    #             nn.Linear(self.n_feature, out_features),
    #             nn.ReLU(True)
    #         )

    def forward(self, x, verbose=False):
        x = self.convencoder(x)

        if verbose:
            print('fusion_conv_out', x.shape)

        x = self.avgpool(x)

        #         x = x.view(-1, self.n_feature)
        x = self.linear(x)

        return x


class FullyConnected(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_features):
        super(FullyConnected, self).__init__()
        self.fc = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),

            nn.Linear(hidden_dim, out_features),
            nn.ReLU(True),

        )

    def forward(self, x, verbose=False):
        x = self.fc(x)
        return x


class TemporalNetwork(nn.Module):

    def __init__(self, in_feature, hidden_feature, out_feature):
        super(TemporalNetwork, self).__init__()

        self.rnn = nn.GRU(in_feature, hidden_feature)

    #         self.fc = nn.Linear(hidden_feature, out_feature)

    def forward(self, sentence):
        rnn_out, _ = self.rnn(sentence)
        #         tag_space = self.fc(rnn_out.view(sentence.shape[0], -1))
        return rnn_out


class BEVNetwork(nn.Module):

    def __init__(self, out_features):
        super(BEVNetwork, self).__init__()
        self.convlayers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=10),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 64, kernel_size=3, padding=10),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, padding=10),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, padding=10),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 128, kernel_size=3, padding=10),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 64, kernel_size=3, padding=10),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, out_features, kernel_size=3, dilation=2)
            # nn.ReLU(),
        )

    def forward(self, x, verbose=False):
        # print('ebv input:', x.shape)
        x = self.convlayers(x)

        if verbose:
            print('ebv_dilated', x.shape)

        x = F.interpolate(x, size=(800, 800), mode='bilinear', align_corners=False)
        return x


class BEVNetworkDeconv(nn.Module):

    def __init__(self, in_features, out_features):
        super(BEVNetworkDeconv, self).__init__()
        self.convlayers = nn.Sequential(
            nn.ConvTranspose2d(in_features, 128, kernel_size=3, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, out_features, kernel_size=3, stride=2)
            # nn.ReLU(),

        )

    def forward(self, x, verbose=False):
        # print('ebv input:', x.shape)
        x = self.convlayers(x)

        if verbose:
            print('ebv_dilated', x.shape)

        x = F.interpolate(x, size=(800, 800), mode='bilinear', align_corners=False)
        return x


class BEVNetworkUnsamp(nn.Module):

    def __init__(self, out_features):
        super(BEVNetworkUnsamp, self).__init__()
        self.convlayers = nn.Sequential(

            # replace ConvTranspose2d with upsample + conv as in paper
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),

            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=out_features, kernel_size=7, padding=0),
        )

    def forward(self, x, verbose=False):
        # print('ebv input:', x.shape)
        x = self.convlayers(x)

        if verbose:
            print('ebv_dilated', x.shape)

        x = F.interpolate(x, size=(800, 800), mode='bilinear', align_corners=False)
        return x


class RoadMapNetwork(nn.Module):
    def __init__(self,
                 single_blocks_sizes=[64, 128, 256],
                 single_depths=[2, 2, 2],
                 fusion_block_sizes=[256, 512],
                 fusion_depths=[2, 2],
                 fusion_out_feature=512,
                 temporal_hidden=512,
                 bev_input_dim=30):
        super(RoadMapNetwork, self).__init__()
        self.bev_input_dim = bev_input_dim

        # single_encoders = {image_num: SingleImageCNN() for image_num in num_images}

        self.single_encoder = SingleImageCNN(
            blocks_sizes=single_blocks_sizes,
            depths=single_depths)

        self.fusion_net = FusionNetwork(
            in_feature=single_blocks_sizes[-1],
            blocks_sizes=fusion_block_sizes,
            depths=fusion_depths,
            out_features=fusion_out_feature)

        self.temporal_net = TemporalNetwork(
            in_feature=fusion_out_feature,
            hidden_feature=temporal_hidden,
            out_feature=bev_input_dim ** 2)

        self.bev_net = BEVNetworkDeconv(
            in_features=bev_input_dim ** 2,
            out_features=1)

    def forward(self, single_cam_input, verbose=False):
        encoder_outputs = []
        for cam_input in single_cam_input:
            output = self.single_encoder(cam_input, verbose)
            encoder_outputs.append(output)

        x = torch.cat(
            [torch.cat(encoder_outputs[:3], dim=-1),
             torch.cat(encoder_outputs[3:], dim=-1)], dim=-2)

        if verbose:
            print('concat_single', x.shape)

        x = self.fusion_net(x, verbose)

        if verbose:
            print('fusion_output', x.shape)

        x = x.view(x.shape[0], 1, x.shape[1])

        x = self.temporal_net(x)

        if verbose:
            print('temporal_output', x.shape)

        x = x.view(x.shape[0], x.shape[2], 1, 1)
        x = self.bev_net(x, verbose)

        if verbose:
            print('bev_output', x.shape)
        return x


class UNetRoadMapNetwork(nn.Module):
    def __init__(self,
                 single_blocks_sizes=[16, 32, 64],
                 single_depths=[2, 2, 2],
                 unet_start_filts=64,
                 unet_depth=5):
        super(UNetRoadMapNetwork, self).__init__()
        #         self.bev_input_dim = bev_input_dim

        self.single_encoder = SingleImageCNN(
            blocks_sizes=single_blocks_sizes,
            depths=single_depths)

        #         self.single_encoder = {image_idx: SingleImageCNN(
        #             blocks_sizes=single_blocks_sizes,
        #             depths=single_depths) for image_idx in range(6)}

        #         for i in range(len(self.single_encoder)):
        #             self.add_module('single_encoder_{}'.format(i), self.single_encoder[i])

        #         self.fusion_net = FusionNetwork(
        #             in_feature=single_blocks_sizes[-1],
        #             blocks_sizes=fusion_block_sizes,
        #             depths=fusion_depths,
        #             out_features=bev_input_dim ** 2)

        #         self.bev_net = BEVNetworkDeconv(
        #             in_features=bev_input_dim ** 2,
        #             out_features=1)

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


class UNetRoadMapNetwork_extend(nn.Module):
    def __init__(self,
                 single_blocks_sizes=[16, 32, 64],
                 single_depths=[2, 2, 2],
                 unet_start_filts=64,
                 unet_depth=5):
        super(UNetRoadMapNetwork_extend, self).__init__()

        self.single_encoder = SingleImageCNN(
            blocks_sizes=single_blocks_sizes,
            depths=single_depths)

        self.u_net = UNet(num_classes=unet_start_filts,
                          in_channels=single_blocks_sizes[-1],
                          depth=unet_depth,
                          start_filts=unet_start_filts, up_mode='transpose',
                          merge_mode='concat')
        self.connect = nn.Sequential(
            nn.ReLU(True),
        )
        self.bev_decode = BEVNetworkDeconv(
            in_features=unet_start_filts,
            out_features=1)

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

        x = self.connect(x)
        x = self.bev_decode(x, verbose)

        #         x = F.interpolate(x, size=(800, 800), mode='bilinear', align_corners=False)

        if verbose:
            print('model end', x.shape)

        return x


class UNetRoadMapNetwork_extend2(nn.Module):
    def __init__(self,
                 single_blocks_sizes=[16, 32, 64],
                 single_depths=[2, 2, 2],
                 unet_start_filts=64,
                 unet_depth=5):
        super(UNetRoadMapNetwork_extend2, self).__init__()
        #         self.bev_input_dim = bev_input_dim

        # TODO: try different models for each camera angle
        # single_encoders = {image_num: SingleImageCNN() for image_num in num_images}

        self.single_encoder = SingleImageCNN(
            blocks_sizes=single_blocks_sizes,
            depths=single_depths)

        #         self.single_encoder = {image_idx: SingleImageCNN(
        #             blocks_sizes=single_blocks_sizes,
        #             depths=single_depths) for image_idx in range(6)}

        #         for i in range(len(self.single_encoder)):
        #             self.add_module('single_encoder_{}'.format(i), self.single_encoder[i])
        self.fusion = nn.Sequential(
            nn.Conv2d(single_blocks_sizes[-1], single_blocks_sizes[-1], kernel_size=1),
            nn.ReLU(True),
        )

        #         self.fusion_net = FusionNetwork(
        #             in_feature=single_blocks_sizes[-1],
        #             blocks_sizes=fusion_block_sizes,
        #             depths=fusion_depths,
        #             out_features=bev_input_dim ** 2)

        #         self.bev_net = BEVNetworkDeconv(
        #             in_features=bev_input_dim ** 2,
        #             out_features=1)

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
        x = self.fusion(x)
        if verbose:
            print('fusion_out', x.shape)

        x = self.u_net(x, verbose)

        if verbose:
            print('unet_output', x.shape)

        x = F.interpolate(x, size=(800, 800), mode='bilinear', align_corners=False)

        if verbose:
            print('interpolate', x.shape)

        return x
    

# Add decoder here, note that the static decoder and dynamic decoder share the same structure

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
    

class Decoder(nn.Module):
    def __init__(self, out_features=1):
        super(Decoder, self).__init__()
        self.num_output_channels = out_features
        # self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 64, 256, 1024, 2048])
        # self.num_ch_concat = np.array([64, 128, 256, 512, 128])
        # self.conv_mu = nn.Conv2d(128, 128, 3, 1, 1)
        # self.conv_log_sigma = nn.Conv2d(128, 128, 3, 1, 1)
        outputs = {}
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = 2048 if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            # num_ch_concat = self.num_ch_concat[i]
            self.convs[("upconv", i, 0)] = nn.Conv2d(num_ch_in, num_ch_out, 3, 1, 1) #Conv3x3(num_ch_in, num_ch_out)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] =  nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(num_ch_out, num_ch_out, 3, 1, 1) #ConvBlock(num_ch_out, num_ch_out)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)

        self.convs["topview"] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        self.dropout = nn.Dropout3d(0.2)
        self.decoder = nn.ModuleList(list(self.convs.values()))



    def forward(self, x, is_training=True, verbose=False):
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
        if verbose:
            print('decode_output', x.shape)

        x = F.interpolate(x, size=(800, 800), mode='bilinear', align_corners=False)
        return x

# Encoder class (changed to deal with 6 camera images)

class Encoder(nn.Module):
    def __init__(self, num_layers, img_ht, img_wt, pretrained=True):
        super(Encoder, self).__init__()

        self.resnet_encoder = ResnetEncoder(num_layers, pretrained)#opt.weights_init == "pretrained"))
        num_ch_enc = self.resnet_encoder.num_ch_enc
        #convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(num_ch_enc[-1], 512)
        self.conv2 = Conv3x3(512, 1024)
        self.pool = nn.MaxPool2d(2)

        #fully connected
        #curr_h = img_ht//(2**6)
        #curr_w = img_wt//(2**6)
        #features_in = curr_h*curr_w*128
        #self.fc_mu = torch.nn.Linear(features_in, 2048)
        #self.fc_sigma = torch.nn.Linear(features_in, 2048)
        #self.fc = torch.nn.Linear(features_in, 2048)
        

    def forward(self, single_cam_input, is_training= True, verbose = False):

        #batch_size, c, h, w = x.shape
        encoder_outputs = []
        for cam_input in single_cam_input:
            output = self.resnet_encoder(cam_input)[-1]
            encoder_outputs.append(output)

        x = torch.cat(
            [torch.cat(encoder_outputs[:3], dim=-1),
             torch.cat(encoder_outputs[3:], dim=-1)], dim=-2)
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = self.pool(x)
        if verbose:
            print("After encoder", x.shape)
            
        return x

