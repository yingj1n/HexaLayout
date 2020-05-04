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
from model import RoadMapEncoder, UNetRoadMapNetwork, UNetRoadMapNetwork_extend, UNetRoadMapNetwork_extend2
import utils
import module_monolayout

import code.data_helper as data_helper
from code.data_helper import UnlabeledDataset, LabeledDataset
from code.helper import collate_fn, draw_box

parser = argparse.ArgumentParser()
parser.add_argument('--folder_dir', type=str, default='./')
parser.add_argument('--verbose_dim', action='store_true')
parser.add_argument('--train_batch_size', type=int, default=9)
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
# val_index_set = np.array([111, 116, 120])

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

# model = LWRoadMapNetwork(
#     single_blocks_sizes=[64, 128, 256],
#     single_depths=[1, 1, 1],
#     fusion_block_sizes=[256, 512, 1024],
#     fusion_depths=[1, 1, 1],
#     fusion_out_feature=2048,
#     temporal_hidden=2048,
#     bev_input_dim=50
# ).to(DEVICE)

models = {'encoder': RoadMapEncoder(
    single_blocks_sizes=[16, 128, 256],
    single_depths=[2, 2, 2]),
    'static_decoder': module_monolayout.Decoder(1),
    'static_discr': module_monolayout.Discriminator(),
    'dynamic_discr': module_monolayout.Discriminator(),
    'dynamic_decoder': module_monolayout.Decoder(10)}

criterion_dynamic = nn.CrossEntropyLoss(weight=torch.FloatTensor([1] + [15] * 9).to(DEVICE))
criterion_static = nn.BCEWithLogitsLoss()

parameters_discr = []
parameters_other = []
for key in models.keys():
    models[key].to(DEVICE)
    if "discr" in key:
        parameters_discr += list(models[key].parameters())
    else:
        parameters_other += list(models[key].parameters())

optimizer_discr = torch.optim.Adam(parameters_discr,
                                   lr=learning_rate,
                                   weight_decay=1e-5)
optimizer_other = torch.optim.Adam(parameters_other,
                                   lr=learning_rate,
                                   weight_decay=1e-5)

# patch = (1, 800 // 2**4, 800 // 2**4)
#
# valid = Variable(torch.Tensor(np.ones((train_batch_size, *patch))),
#                                              requires_grad=False).float().to(DEVICE)
# fake  = Variable(torch.Tensor(np.zeros((train_batch_size, *patch))),
#                                              requires_grad=False).float().to(DEVICE)


## Training Loop
learning_curve = []
# model.to(DEVICE)
val_ts_list = []
for epoch in range(num_epochs):
    train_loss = 0
    train_ts_list = []
    sample_size = 0
    start_time = time.time()
    batch_end_time = start_time
    # model.train()
    # model = model.to(DEVICE)
    models = utils.to_train(models, DEVICE)
    for batch, (sample, target, road_image, extra) in enumerate(train_loader):
        batch_size = len(sample)
        target_bb_map = torch.stack([utils.bounding_box_to_matrix_image(i) for i in target]).to(DEVICE)
        road_image_long = torch.stack(road_image).type(torch.FloatTensor).to(DEVICE)

        single_cam_inputs = []
        for i in range(num_images):
            single_cam_input = torch.stack([batch[i] for batch in sample])
            #             single_cam_input = utils.images_transform(single_cam_input, i)
            single_cam_input = Variable(single_cam_input).to(DEVICE)
            single_cam_inputs.append(single_cam_input)

        # ===================forward=====================
        encoded_features = models['encoder'](single_cam_inputs, opt.verbose_dim)
        outputs = {}
        outputs["dynamic"] = models["dynamic_decoder"](encoded_features)
        outputs["static"] = models["static_decoder"](encoded_features)

        if opt.verbose_dim:
            print(outputs["dynamic"].shape, outputs["static"].shape)

        loss_dynamic = criterion_dynamic(outputs["dynamic"], target_bb_map)
        loss_static = criterion_static(outputs["static"].view(-1, 800, 800), road_image_long)
        loss = loss_static + loss_dynamic

        # ===================backward====================
        optimizer_other.zero_grad()
        loss.backward()
        optimizer_other.step()

        # ===================log========================
        train_loss += loss.item() * batch_size
        sample_size += batch_size
        predicted_road_map = (outputs["static"] > 0.5).view(-1, 800, 800)
        # predicted_bb_map = np.argmax(outputs["dynamic"].cpu().detach().numpy(), axis=1)
        # TODO: turn bb map to bbox and measure ts


        batch_ts, _ = utils.get_ts_for_batch_binary(outputs["static"], road_image)
        train_ts_list.extend(batch_ts)
        avg_train_ts = sum(batch_ts) / len(batch_ts)

        batch_time = time.time() - batch_end_time
        batch_end_time = time.time()
        if batch % 10 == 0:
            print('batch [{}], epoch [{}], loss: {:.4f}, time: {:.0f}s, ts: {:.4f}'
                  .format(batch + 1, epoch + 1, train_loss / sample_size,
                          batch_time, avg_train_ts))

        # Empty Cache
        torch.cuda.empty_cache()

    # ===================log every epoch======================
    train_ts = sum(train_ts_list) / len(train_ts_list)
    val_ts, predicted_val_map = utils.evaluation_layout(models, val_loader, DEVICE)
    time_this_epoch_min = (time.time() - start_time) / 60
    print('epoch [{}/{}], loss: {:.4f}, time: {:.2f}min, remaining: {:.2f}min, train_ts: {:.4f}, val_ts: {:.4f}'
          .format(epoch + 1, num_epochs, train_loss / sample_size,
                  time_this_epoch_min, time_this_epoch_min * (num_epochs - epoch - 1), train_ts, val_ts))

    learning_curve.append((train_ts, val_ts.tolist()))
    if epoch % 5 == 0 and epoch != 0:
        torch.save({
            'encoder': models['encoder'].state_dict(),
            'static_decoder': models['static_decoder'].state_dict(),
            'dynamic_decoder': models['dynamic_decoder'].state_dict(),
            'plot_cache': learning_curve,
            'predicted_val_map': predicted_val_map
        },
            FOLDER_PATH + '/roadmap_models/roadmapnet_{}_{}.pth'.format(timestampStr, epoch))

torch.save({
    'encoder': models['encoder'].state_dict(),
    'static_decoder': models['static_decoder'].state_dict(),
    'dynamic_decoder': models['dynamic_decoder'].state_dict(),
    'plot_cache': learning_curve,
    'predicted_val_map': predicted_val_map
},
    FOLDER_PATH + '/roadmap_models/roadmapnet_{}_final.pth'.format(timestampStr))
