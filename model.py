import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from module_unet import UNet
from module_resnet import ResNetEncoder


class SingleImageCNN(nn.Module):
    def __init__(self, blocks_sizes=[64, 128, 256], depths=[2, 2, 2], in_feature=3):
        super(SingleImageCNN, self).__init__()
        self.convencoder = ResNetEncoder(in_feature, blocks_sizes=blocks_sizes, depths=depths)

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

#         x = self.avgpool(x)

#         #         x = x.view(-1, self.n_feature)
#         x = self.linear(x)

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


class RoadMapEncoder(nn.Module):
    def __init__(self,
                 single_in_feature=3,
                 single_blocks_sizes=[16, 32],
                 single_depths=[2, 2],
                 fusion_block_sizes=[256, 512],
                 fusion_depths=[2, 2],
                 fusion_out_feature=512,
                ):
        super(RoadMapEncoder, self).__init__()
        #         self.bev_input_dim = bev_input_dim

        # TODO: try different models for each camera angle
        # single_encoders = {image_num: SingleImageCNN() for image_num in num_images}

        self.single_encoder = SingleImageCNN(
            blocks_sizes=single_blocks_sizes,
            depths=single_depths,
            in_feature=single_in_feature)

        #         self.single_encoder = {image_idx: SingleImageCNN(
        #             blocks_sizes=single_blocks_sizes,
        #             depths=single_depths) for image_idx in range(6)}

        #         for i in range(len(self.single_encoder)):
        #             self.add_module('single_encoder_{}'.format(i), self.single_encoder[i])
        self.fusion = FusionNetwork(
            in_feature=single_blocks_sizes[-1],
            blocks_sizes=fusion_block_sizes,
            depths=fusion_depths,
            out_features=fusion_out_feature)


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

        return x
    
    
class RoadMapEncoder_temporal(nn.Module):
    def __init__(self,
                 single_in_feature=3,
                 single_blocks_sizes=[64, 128, 256],
                 single_depths=[2, 2, 2],
                 fusion_block_sizes=[256, 512],
                 fusion_depths=[2, 2],
                 fusion_out_feature=512,
                 temporal_hidden=512,
                 output_size=512):
        super(RoadMapEncoder_temporal, self).__init__()

        self.single_encoder = SingleImageCNN(
            blocks_sizes=single_blocks_sizes,
            depths=single_depths,
            in_feature=single_in_feature)

        self.fusion_net = FusionNetwork(
            in_feature=single_blocks_sizes[-1],
            blocks_sizes=fusion_block_sizes,
            depths=fusion_depths,
            out_features=fusion_out_feature)

        self.temporal_net = TemporalNetwork(
            in_feature=fusion_out_feature,
            hidden_feature=temporal_hidden,
            out_feature=output_size)


    def forward(self, single_cam_input, verbose=False):
        encoder_outputs = []
        for cam_input in single_cam_input:
            output = self.single_encoder(cam_input, verbose)
            encoder_outputs.append(output)

        x = utils.combine_six_to_one(encoder_outputs)

        if verbose:
            print('concat_single', x.shape)

        x = self.fusion_net(x, verbose)

        if verbose:
            print('fusion_output', x.shape)

        x = torch.transpose(x, 1, 2)
        x = torch.unbind(x, -1)[0]
#         x = x.view(x.shape[0], 1, x.shape[1])  # turns out '.view' messes up the data distribution
        if verbose:
            print('reshaped', x.shape)

        x = self.temporal_net(x)

        if verbose:
            print('temporal_output', x.shape)

        x = torch.transpose(x, 1, 2)
        x = torch.unsqueeze(x, 3)
        if verbose:
            print('reshaped', x.shape)
#         x = x.view(x.shape[0], x.shape[2], 1, 1)
        return x
