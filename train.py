# import os
import random
import time
import argparse

import numpy as np
# import pandas as pd
from datetime import datetime
import pytz

# import matplotlib
# import matplotlib.pyplot as plt
#
# matplotlib.rcParams['figure.figsize'] = [5, 5]
# matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

# from model import RoadMapNetwork, LWRoadMapNetwork
from model import UNetRoadMapNetwork, UNetRoadMapNetwork_extend, UNetRoadMapNetwork_extend2
import utils

import code.data_helper as data_helper
from code.data_helper import UnlabeledDataset, LabeledDataset
from code.helper import collate_fn, draw_box
from code.helper import compute_iou, compute_ats_bounding_boxes

parser = argparse.ArgumentParser()
parser.add_argument('--folder_dir', type=str, default='./')
parser.add_argument('--verbose_dim', action='store_true')
parser.add_argument('--train_batch_size', type=int, default=9)
parser.add_argument('--train_object_detection', type=bool, default = True)
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
val_index_set = np.array([111, 116, 120])

# transform = torchvision.transforms.ToTensor()
# transform = torchvision.transforms.Compose(
#     [torchvision.transforms.ToTensor(),
#      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                       std=[0.229, 0.224, 0.225])])

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                      std=(0.5, 0.5, 0.5))])

num_images = data_helper.NUM_IMAGE_PER_SAMPLE
train_batch_size = opt.train_batch_size
val_batch_size = 1
learning_rate = 0.001
num_epochs = 200

# The labeled dataset can only be retrieved by sample.
# And all the returned data are tuple of tensors, since bounding boxes may have different size
# You can choose whether the loader returns the extra_info. It is optional. You don't have to use it.
labeled_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=train_index_set,
                                  transform=transform,
                                  extra_info=True
                                  )
# train_loader = torch.utils.data.DataLoader(
#     labeled_trainset,
#     batch_sampler=utils.RandomBatchSampler(
#         sampler=torch.utils.data.SequentialSampler(labeled_trainset),
#         batch_size=train_batch_size),
#     num_workers=0, collate_fn=collate_fn)

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
from tqdm import tqdm
for sample, target, road_image, extra in tqdm(val_loader):
    print(len(target))
    torch.tensor(utils.bounding_box_to_matrix_image(target[0])).to('cpu')
    print('done')
    break 
print('done2')
# model = LWRoadMapNetwork( 
#     single_blocks_sizes=[64, 128, 256],
#     single_depths=[1, 1, 1],
#     fusion_block_sizes=[256, 512, 1024],
#     fusion_depths=[1, 1, 1],
#     fusion_out_feature=2048,
#     temporal_hidden=2048,
#     bev_input_dim=50
# ).to(DEVICE)

################################## SPECIFY YOUR MODEL HERE ##############################
if not opt.train_object_detection: # original lane dection model 
    model = UNetRoadMapNetwork_extend2(
            single_blocks_sizes=[16, 64],
                    single_depths=[2, 2],
                    unet_start_filts=64,
                    unet_depth=5

    ).to(DEVICE)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate,
                                weight_decay=1e-5)


else: 
    # model structure is basically same 
    model = UNetRoadMapNetwork(
            single_blocks_sizes=[16, 64],
                    single_depths=[2, 2],
                    unet_start_filts=64,
                    unet_depth=5

    ).to(DEVICE)

    criterion_bbox = nn.CrossEntropyLoss()
    criterion_lane = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate,
                                weight_decay=1e-5)


#########################################################################################



## Training Loop
learning_curve = []
model.to(DEVICE)
val_ts_list = []
for epoch in range(num_epochs):
    train_ts_list = []
    train_lane_loss = 0
    #-----------------
    # FOR object detection 
    train_iou_list = []
    train_loss_bbox = 0
    #----------------
    sample_size = 0
    start_time = time.time()
    batch_end_time = start_time
    model.train()
    model = model.to(DEVICE)
    for batch, (sample, target, road_image, extra) in enumerate(train_loader):
        batch_size = len(sample)

        single_cam_inputs = []
        for i in range(num_images):
            single_cam_input = torch.stack([batch[i] for batch in sample])
#             single_cam_input = utils.images_transform(single_cam_input, i)
            single_cam_input = Variable(single_cam_input).to(DEVICE)
            single_cam_inputs.append(single_cam_input)

        ######## Change here so that model produces to output and back propagate to the same feature extractor #############
        # ===================forward=====================
        bev_lane_output = model(single_cam_inputs, True, opt.verbose_dim)
        bev_bbox_output = model(single_cam_inputs, False, opt.verbose_dim)
        #print('original bbox output shape',bev_bbox_output.shape) # torch.Size([9, 10, 800, 800])
        _, bev_bbox_pred = torch.max(bev_bbox_output, dim = 1)
        road_image_long = torch.stack(road_image).type(torch.FloatTensor).to(DEVICE) # torch.Size([9, 800, 800]) 

        bbox_matrix_long = torch.tensor([utils.bounding_box_to_matrix_image(i) for i in target], dtype = torch.long).to(DEVICE) # torch.Size([9, 800, 800])
        # print('bbox matrices ts size',  bbox_matrix_long.shape) # torch.Size([9, 800, 800])
        
        # print(bev_lane_output.shape, road_image_long.shape)
        loss_lane = criterion_lane(bev_lane_output.view(-1, 800, 800),
                         road_image_long)
        
        loss_bbox = criterion_bbox(bev_bbox_output, bbox_matrix_long)
        total_loss = loss_lane + loss_bbox

        # ===================backward====================
        optimizer.zero_grad()
        
        #loss_lane.backward()
        total_loss.backward()
        optimizer.step()

        # ===================log========================
        train_lane_loss += loss_lane.item() * batch_size
        sample_size += batch_size
#         predicted_road_map = np.argmax(bev_lane_output.cpu().detach().numpy(), axis=1).astype(bool)
        predicted_road_map = (bev_lane_output > 0.5).view(-1, 800, 800)
#         print(predicted_road_map.shape, road_image[0].shape)
#         print(road_image)

        # ----- bbox
        train_loss_bbox += loss_bbox.item() * batch_size
        train_iou = utils.compute_bbox_matrix_iou(labels = bbox_matrix_long, 
                                                predictions = bev_bbox_pred)
        train_iou_list.append(train_iou)
        # -----
        batch_ts, _ = utils.get_ts_for_batch_binary(bev_lane_output, road_image)
        train_ts_list.extend(batch_ts)
        avg_train_ts = sum(batch_ts) / len(batch_ts)

        batch_time = time.time() - batch_end_time
        batch_end_time = time.time()
        if batch % 10 == 0:
            print('batch [{}], epoch [{}], lane_loss: {:.4f}, bbox_loss: {:.4f},time: {:.0f}s, ts_lane: {:.4f}, mean_iou: {:.4f}'
                  .format(batch + 1, epoch + 1, train_lane_loss / sample_size, train_loss_bbox / sample_size, 
                          batch_time, avg_train_ts, np.average(train_iou_list)))

        # Empty Cache
        torch.cuda.empty_cache()

    # ===================log every epoch======================
   #print(train_ts_list, len(train_ts_list))
    train_ts = sum(train_ts_list) / len(train_ts_list) 
    val_ts, predicted_val_map, val_iou_list = utils.evaluation(model, val_loader, DEVICE)
    time_this_epoch_min = (time.time() - start_time)/60
    print('epoch [{}/{}], loss: {:.4f}, time: {:.2f}min, remaining: {:.2f}min,\
     train_ts: {:.4f}, val_ts: {:.4f}, train_iou: {:.4f}, val_iou: {:.4f}'
          .format(epoch + 1, num_epochs, train_lane_loss / sample_size,
                  time_this_epoch_min, time_this_epoch_min*(num_epochs-epoch-1), train_ts, val_ts,
                np.average(train_iou_list), np.average(val_iou)))

    learning_curve.append((train_ts, val_ts.tolist()))
    if epoch % 5 == 0 and epoch != 0:
        torch.save({
            'state_dict': model.state_dict(),
            'plot_cache': learning_curve,
            'predicted_val_map': predicted_val_map
        },
            FOLDER_PATH + '/roadmap_models/roadmapnet_{}_{}.pth'.format(timestampStr, epoch))

torch.save({
    'state_dict': model.state_dict(),
    'plot_cache': learning_curve,
    'predicted_val_map': predicted_val_map
},
    FOLDER_PATH + '/roadmap_models/roadmapnet_{}_final.pth'.format(timestampStr))
