import os
import random
import time
import argparse

import numpy as np
# import pandas as pd
from datetime import datetime
import pytz

import PIL.Image as pil
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

import module_monodepth2 # Source from monodepth2
from layers import disp_to_depth # Source from monodepth2

from road_map_model import RoadMapEncoder, RoadMapEncoder_temporal
from model import UnetDecoder
#from mono_model import Decoder, Encoder
import utils

import code.data_helper as data_helper
from code.data_helper import UnlabeledDataset, LabeledDataset
from code.helper import collate_fn, draw_box, compute_ats_bounding_boxes

parser = argparse.ArgumentParser()
parser.add_argument('--folder_dir', type=str, default='./')
parser.add_argument('--depth_model_dir', type=str, default='./')
parser.add_argument('--verbose_dim', action='store_true')
parser.add_argument('--train_batch_size', type=int, default=9)
parser.add_argument('--bbox_label', action='store_true')
parser.add_argument('--load_model', action='store_true')
parser.add_argument('--model_path', type=str, default='./')

opt = parser.parse_args()

random.seed(888)
np.random.seed(888)
torch.manual_seed(888)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on {}'.format(DEVICE))

FOLDER_PATH = opt.folder_dir
image_folder = FOLDER_PATH + 'data'
annotation_csv = FOLDER_PATH + 'data/annotation.csv'

now = datetime.now()
timezone = pytz.timezone("America/Los_Angeles")
now_la = timezone.localize(now)
timestampStr = now_la.strftime("%m-%d-%H-%M")

# unlabeled_scene_index = np.arange(106)
labeled_scene_index = np.arange(106, 134)

train_index_set = np.array(
    [133, 118, 130, 119, 107, 114, 122, 121, 132, 115, 126, 117, 112, 128, 108, 110, 131, 129, 124, 125, 106, 109])
# small_train_index_set = np.array([133, 118, 130, 119, 107, 114, 122, 121, 132])
val_index_set = np.array([i for i in labeled_scene_index if i not in set(train_index_set)])
# print(val_index_set) [106 109 111 113 116 120 123 127]
#val_index_set = np.array([111, 116, 120])

# transform = torchvision.transforms.ToTensor()
# transform = torchvision.transforms.Compose(
#     [torchvision.transforms.ToTensor(),
#      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                       std=[0.229, 0.224, 0.225])])

transform = torchvision.transforms.Compose(
   [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))])

#transform = torchvision.transforms.ToTensor()

num_images = data_helper.NUM_IMAGE_PER_SAMPLE
train_batch_size = opt.train_batch_size
val_batch_size = 1
learning_rate = 0.001
num_epochs = 200
depth_width = 320
depth_height = 256
original_width = 306
original_height = 256

# The labeled dataset can only be retrieved by sample.
# And all the returned data are tuple of tensors, since bounding boxes may have different size
# You can choose whether the loader returns the extra_info. It is optional. You don't have to use it.
labeled_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=train_index_set,
                                  transform=transform,
                                  extra_info=True
                                  )


train_loader = torch.utils.data.DataLoader(labeled_trainset,
                                         batch_size=train_batch_size,
                                         shuffle=True, num_workers=0,
                                         collate_fn=collate_fn)

labeled_valset = LabeledDataset(image_folder=image_folder,
                                annotation_file=annotation_csv,
                                scene_index=val_index_set,
                                transform=transform,
                                extra_info=True
                                )
val_loader = torch.utils.data.DataLoader(labeled_valset,
                                         batch_size=val_batch_size,
                                         shuffle=False, num_workers=0,
                                         collate_fn=collate_fn)

if opt.bbox_label:
    bbox_out_features = 10
    criterion_dynamic = nn.CrossEntropyLoss()
else:
    bbox_out_features = 2
    criterion_dynamic = nn.CrossEntropyLoss()
    #criterion_dynamic = nn.BCEWithLogitsLoss()

#criterion_static = nn.BCEWithLogitsLoss()
criterion_static = nn.CrossEntropyLoss()

#models = {'raw_encoder': Encoder(18, 256, 306, False),
#          'depth_encoder': Encoder(18, 256, 306, False),
#          'static_decoder': Decoder(out_features=1),
#          'dynamic_decoder': Decoder(out_features=bbox_out_features)}

blocks_sizes = [16, 128, 256, 512, 2048]
single_blocks_sizes=[64, 128, 256] # Add for unet

models = {'raw_encoder': RoadMapEncoder(
                         single_in_feature=3,
                         single_blocks_sizes=[64, 128, 256, 512],
                         single_depths=[2, 2, 2], #[2,2,2,2] initially, changed for testing unet
                         fusion_block_sizes=[512, 1024],
                         fusion_depths=[1, 1],
                         fusion_out_feature=1024,
                         fusion_on = False),
          'depth_encoder': RoadMapEncoder(
                         single_in_feature=1,
                         single_blocks_sizes=[64, 128, 256, 512],
                         single_depths=[2, 2, 2], #[2,2,2,2] initially, changed for testing unet
                         fusion_block_sizes=[512, 1024],
                         fusion_depths=[1, 1],
                         fusion_out_feature=1024,
                         fusion_on = False),
        #   'static_decoder': Decoder(
        #                  blocks_sizes=blocks_sizes,
        #                  out_features=1),
        #   'dynamic_decoder': Decoder(
        #                  blocks_sizes=blocks_sizes,
        #                  out_features=bbox_out_features)
        'static_decoder': UnetDecoder(single_block_size_output = single_blocks_sizes[-1]*2, 
                                num_objects = 2),
        'dynamic_decoder': UnetDecoder(single_block_size_output = single_blocks_sizes[-1]*2, 
                                num_objects = bbox_out_features,
                                )}

if opt.load_model:
    checkpoint = torch.load(opt.model_path)
    models['raw_encoder'].load_state_dict(checkpoint['raw_encoder'])
    models['depth_encoder'].load_state_dict(checkpoint['depth_encoder'])
    models['static_decoder'].load_state_dict(checkpoint['static_decoder'])
    models['dynamic_decoder'].load_state_dict(checkpoint['dynamic_decoder'])

# Add depth model
depth_model_path = opt.depth_model_dir
model_name = "weights_19"
dp_model_list = ["front_left_savings/mdp", "front_savings/mdp", "front_right_savings/mdp",
                 "back_left_savings/mdp", "back_savings/mdp", "back_right_savings/mdp"]
encoder_path_list = []
decoder_path_list = []

for i in range(6):
    encoder_path_list.append(os.path.join(depth_model_path, dp_model_list[i], "models", model_name, "encoder.pth"))
    decoder_path_list.append(os.path.join(depth_model_path, dp_model_list[i], "models", model_name, "depth.pth"))

# LOADING PRETRAINED MODEL
encoder_model_list = [module_monodepth2.ResnetEncoder(18, False) for i in range(6)]
depth_decoder_model_list = [module_monodepth2.DepthDecoder(num_ch_enc=encoder_model_list[0].num_ch_enc, scales=range(4)) for i in range(6)]
#camera_keys = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

# Get a dictionary to save model for use later
#encoder_models = {}
#depth_decoder_models = {}

for i in range(6):
    loaded_dict_enc = torch.load(encoder_path_list[i])
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder_model_list[i].state_dict()}
    encoder_model_list[i].load_state_dict(filtered_dict_enc)
    encoder_model_list[i].to(DEVICE)
    encoder_model_list[i].eval()
    # encoder_models[camera_keys[i]] = encoder_model_list[i]

    # Decoder
    loaded_dict = torch.load(decoder_path_list[i])
    depth_decoder_model_list[i].load_state_dict(loaded_dict)
    depth_decoder_model_list[i].to(DEVICE)
    depth_decoder_model_list[i].eval()
    # depth_decoder_models[camera_keys[i]] = depth_decoder_model_list[i]

parameters_other = []
for key in models.keys():
    models[key].to(DEVICE)
    if "discr" in key:
        pass
    else:
        parameters_other += list(models[key].parameters())

optimizer_other = torch.optim.Adam(parameters_other,
                             lr=learning_rate,
                             weight_decay=1e-5)

## Training Loop
learning_curve = []
# model.to(DEVICE)
val_ts_list = []
for epoch in range(num_epochs):
    train_loss = 0
    train_ts_list = []
    train_bb_ts_list = []
    sample_size = 0
    start_time = time.time()
    batch_end_time = start_time

    ## Add some evaluation here
    #if opt.load_model:
    #    val_ts, val_bscore, accu_auc = utils.evaluation_layout(models, encoder_model_list, depth_decoder_model_list, 
                                        val_loader, DEVICE)
    #    print('Evaluation result for model [{}], val_ts: {:.4f}, val_bscore: {:.4f}, road_map acc: {:.4f}, road_map auc: {:.4f}, bbox acc: {:.4f}, bbox auc: {:.4f}'
    #      .format(opt.model_path.split("/")[-1], val_ts, val_bscore, accu_auc["rm"][0], accu_auc["rm"][1], accu_auc["bb"][0], accu_auc["bb"][1]))
    #break # To delete when training

    models = utils.to_train(models, DEVICE)
    for batch, (sample, target, road_image, extra) in enumerate(train_loader):
        batch_size = len(sample)
        target_bb_map = torch.stack([utils.bounding_box_to_matrix_image(i, opt.bbox_label) for i in target]).to(DEVICE)
        road_image_long = torch.stack(road_image).type(torch.LongTensor).to(DEVICE)
        #road_image_long = torch.stack(road_image).type(torch.FloatTensor).to(DEVICE)
        single_cam_inputs = []
        raw_depth_inputs = [] 
        for i in range(num_images):
            single_cam_input = torch.stack([image[i] for image in sample])
            # Need to get depth outputs as well, since the required input size for depth model is 320 * 256,
            # need to do some processing
            input_for_depth_list = []
            for j in range(batch_size):
                im = single_cam_input[j]
                # de-transform
                im = im * 0.5 + 0.5
                pil_im = torchvision.transforms.ToPILImage()(im)
                pil_im = pil_im.resize((depth_width, depth_height), pil.LANCZOS)
                input_for_depth_list.append(transform(pil_im))
            input_for_depth = torch.stack(input_for_depth_list).to(DEVICE)
            # Get outputs of depth model
            with torch.no_grad():
                d_features = encoder_model_list[i](input_for_depth)
                disp_outputs = depth_decoder_model_list[i](d_features)[("disp", 0)]
            # We need to change the outputs dimension back to original dimension
            disp_resized = nn.functional.interpolate(disp_outputs, (original_height, original_width), mode="bilinear", align_corners=False)
            # Now we need to transfer the output disparity map to depth map
            _, depth_out = disp_to_depth(disp_resized, 1, 40) # We set min depth to be 1 and max depth be 40
            # Reproduce the only channel 3 times for it to match with encoder
            #depth_out = torch.cat([depth_out, depth_out, depth_out], 1)
            
            raw_depth_inputs.append(depth_out)
            
            #             single_cam_input = utils.images_transform(single_cam_input, i)
            single_cam_input = Variable(single_cam_input).to(DEVICE)
            single_cam_inputs.append(single_cam_input)

        # ===================forward=====================
        encoded_raw_features = models['raw_encoder'](single_cam_inputs, verbose = opt.verbose_dim)
        encoded_depth_features = models['depth_encoder'](raw_depth_inputs, verbose = opt.verbose_dim)
        
        # Now we concat the features from raw input and depth input
        encoded_features = torch.cat([encoded_raw_features, encoded_depth_features], 1)
        outputs = {}
        outputs["dynamic"] = models["dynamic_decoder"](encoded_features)
        outputs["static"] = models["static_decoder"](encoded_features)

        if opt.verbose_dim:
            print(outputs["dynamic"].shape, outputs["static"].shape)
        
        if opt.bbox_label:
            loss_dynamic = criterion_dynamic(outputs["dynamic"], target_bb_map)
        else:
            loss_dynamic = criterion_dynamic(outputs["dynamic"], target_bb_map.type(torch.LongTensor).to(DEVICE))
            #loss_dynamic = criterion_dynamic(outputs["dynamic"].view(-1, 800, 800), target_bb_map.type(torch.FloatTensor).to(DEVICE))
        loss_static = criterion_static(outputs["static"], road_image_long) #.view(-1, 800, 800)
        loss = loss_static + 10 * loss_dynamic

        # ===================backward====================
        optimizer_other.zero_grad()
        loss.backward()
        optimizer_other.step()

        # ===================log========================
        train_loss += loss.item() * batch_size
        sample_size += batch_size
        #predicted_road_map = (outputs["static"] > 0.5).view(-1, 800, 800)
        # Add bbox comparison
        #predicted_bb_map = (outputs["dynamic"].view(-1, 800, 800) > 0)
        
        # Get bbox prediction from bb_maps
        # bbox_list = []
        # for i in range(batch_size):
        #     bbox_list.append(utils.image_to_bbox(predicted_bb_map[i]))

        # Road map score
        batch_ts, _ = utils.get_rm_ts_for_batch(outputs["static"], road_image)
        #batch_ts, _ = utils.get_ts_for_batch_binary(outputs["static"], road_image)
        train_ts_list.extend(batch_ts)
        avg_train_ts = sum(batch_ts) / len(batch_ts)

        # bbox score
        batch_bb_ts, _ = utils.get_bb_ts_for_batch(outputs["dynamic"], target)
        train_bb_ts_list.extend(batch_bb_ts)
        avg_train_bb_ts = sum(batch_bb_ts) / len(batch_bb_ts)

        batch_time = time.time() - batch_end_time
        batch_end_time = time.time()
        if batch % 10 == 0:
            print('batch [{}], epoch [{}], loss: {:.4f}, time: {:.0f}s, ts: {:.4f}, bbox_score: {:.4f}'
                  .format(batch + 1, epoch + 1, train_loss / sample_size,
                          batch_time, avg_train_ts, avg_train_bb_ts))

        # Empty Cache
        #torch.cuda.empty_cache()

    # ===================log every epoch======================
    train_ts = sum(train_ts_list) / len(train_ts_list)
    train_bb_ts = sum(train_bb_ts_list) / len(train_bb_ts_list)
    val_ts, val_bscore, accu_acc = utils.evaluation_layout_de(models, encoder_model_list, depth_decoder_model_list, val_loader, DEVICE)
    time_this_epoch_min = (time.time() - start_time) / 60
    print('epoch [{}/{}], loss: {:.4f}, time: {:.2f}min, remaining: {:.2f}min, train_ts: {:.4f}, val_ts: {:.4f}, train_bscore: {:.4f}, val_bscore: {:.4f}'
          .format(epoch + 1, num_epochs, train_loss / sample_size,
                  time_this_epoch_min, time_this_epoch_min * (num_epochs - epoch - 1), train_ts, val_ts, train_bb_ts, val_bscore))

    learning_curve.append((train_ts, val_ts.tolist()))
    if epoch % 5 == 0 and epoch != 0:
        torch.save({
            'raw_encoder': models['raw_encoder'].state_dict(),
            'depth_encoder': models['depth_encoder'].state_dict(),
            'static_decoder': models['static_decoder'].state_dict(),
            'dynamic_decoder': models['dynamic_decoder'].state_dict(),
            'plot_cache': learning_curve,
        },
            FOLDER_PATH + 'roadmap_models_v3/roadmapnet_{}_{}.pth'.format(timestampStr, epoch))

torch.save({
    'raw_encoder': models['raw_encoder'].state_dict(),
    'depth_encoder': models['depth_encoder'].state_dict(),
    'static_decoder': models['static_decoder'].state_dict(),
    'dynamic_decoder': models['dynamic_decoder'].state_dict(),
    'plot_cache': learning_curve,
},
    FOLDER_PATH + '/roadmap_models_v3/roadmapnet_{}_final.pth'.format(timestampStr))
