import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from module_unet import UNet
from module_resnet import ResNetEncoder
from collections import OrderedDict

NUM_OBJECTS = 10
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

##############################################################################
####################### MONO LAYOUT MODELS ###################################
##############################################################################

# encoder: the same 
class RoadEncoder(nn.Module):
    def __init__(self, 
                single_blocks_sizes = [16, 32, 64],
                single_depths = [2, 2, 2], 
                fusion_block_sizes = [256, 512],
                fusion_depths =  [2, 2],
                fusion_out_feature= 512,
                fusion_on = False):

        super(RoadEncoder, self).__init__()

        self.single_encoder = SingleImageCNN(
            blocks_sizes = single_blocks_sizes,
            depths = single_depths,
        )
        # whether to turn on fusion layer, not applicable with 
        # unet decoders, 
        self.fusion_on = fusion_on
        if self.fusion_on: 
            self.fusion = FusionNetwork(
                in_feature=single_blocks_sizes[-1],
                blocks_sizes=fusion_block_sizes,
                depths=fusion_depths,
                out_features=fusion_out_feature)

    def forward(self, single_cam_input,  verbose=False):
        encoder_outputs = []
        for idx, cam_input in enumerate(single_cam_input):
            output = self.single_encoder(cam_input, verbose)
            encoder_outputs.append(output)

        x = utils.combine_six_to_one(encoder_outputs) ### try with other combining methods 

        if verbose:
            print('combined single encoders output', x.shape)
        
        if self.fusion_on: 
            x = self.fusion(x)

            if verbose:
                print('fusion layer output', x.shape)

        return x

# decoder: from mono layout

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


class MonoDecoder(nn.Module):
    def __init__(self, 
                single_block_size_output = 64,
                features = 1):
        super(MonoDecoder, self).__init__()
    
        self.num_output_channels = features 
        self.num_ch_enc = single_block_size_output
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        outputs = {}
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc if i == 4 else self.num_ch_dec[i + 1]
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

    def forward(self, x, verbose = False):
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)
            x = upsample(x)
            x = self.convs[("upconv", i, 1)](x)
            x = self.convs[("norm", i, 1)](x)
       
        x = self.convs["topview"](x) 

        if verbose:
            print('decoder output', x.shape) # torch.Size([9, 2, 3072, 2048])
            
        x = F.interpolate(x, size=(800, 800), mode='bilinear', align_corners=False)

        if verbose:
            print('interpolate', x.shape) # torch.Size([9, 2, 800, 800])
        return x

# decoder: from Unet

class UnetDecoder(nn.Module):
    def __init__(self, 
                single_block_size_output = 64,
                unet_start_filts=64,
                unet_depth=5,
                num_objects=1, 
                ):
        super(UnetDecoder, self).__init__()

        self.u_net = UNet(num_classes=num_objects,
                            in_channels=single_block_size_output,
                            depth=unet_depth,
                            start_filts=unet_start_filts, up_mode='transpose',
                            merge_mode='concat')
        

    def forward(self, encoder_output, verbose = False):
        
        x = self.u_net(encoder_output, verbose)

        if verbose:
            print('decoder output', x.shape)

        x = F.interpolate(x, size=(800, 800), mode='bilinear', align_corners=False)

        if verbose:
            print('interpolate', x.shape)

        return x

    
# discriminator 


class Discriminator(nn.Module):
    def __init__(self, input_channel = 2):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(input_channel, 8, 3, 2, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(8, 16, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(16, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(32, 8, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(8, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, verbose=False):
        x = self.main(input)
        if verbose:
            print('discriminator shape', x.shape)
        #x = F.interpolate(x, size= (800,800), mode='bilinear', align_corners=False )
        return x

#####################################################################################
#####################################################################################
#####################################################################################

