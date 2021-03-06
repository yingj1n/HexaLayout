import random
import time
import argparse
import numpy as np
from datetime import datetime
import pytz
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from model import RoadEncoder, UnetDecoder
import utils
import code.data_helper as data_helper
from code.data_helper import UnlabeledDataset, LabeledDataset
from code.helper import collate_fn, draw_box
from code.helper import compute_iou, compute_ats_bounding_boxes

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--folder_dir', type=str, default='./')
## training 
parser.add_argument('--train_batch_size', type=int, default=9)
parser.add_argument('--num_dynamic_labels', type=int, default=2,
                    choices=[1, 2, 10])
## loss
parser.add_argument("--scheduler_step_size", type=int, default=5,
                         help="step size for the both schedulers")
parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate for both schedulers")
## logs
parser.add_argument("--verbose",
                         help="if print out dimensions in between model steps") 
                                   
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
val_index_set = np.array([i for i in labeled_scene_index if i not in set(train_index_set)])


transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                      std=(0.5, 0.5, 0.5))])

num_images = data_helper.NUM_IMAGE_PER_SAMPLE
train_batch_size = opt.train_batch_size
val_batch_size = 1
num_epochs = 50

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



################################## SPECIFY MODEL HERE ##############################

# define model
models = {}
single_blocks_sizes = [64, 128, 256]
models['encode'] = RoadEncoder(
                    single_blocks_sizes = single_blocks_sizes, ### TODO: try with different state-of-art in-out channels 
                    single_depths = [2,2,2]
                    )
models['static'] = UnetDecoder(single_block_size_output = single_blocks_sizes[-1], 
                                 num_objects = 1) 
models['dynamic'] = UnetDecoder(single_block_size_output = single_blocks_sizes[-1], 
                                 num_objects = opt.num_dynamic_labels)  


# define loss func
criterion_static = nn.BCEWithLogitsLoss()
labels=False
if opt.num_dynamic_labels == 10:   
    criterion_dynamic = nn.CrossEntropyLoss(weight=torch.FloatTensor([1] + [15] * 9).to(DEVICE))
    labels = True
elif opt.num_dynamic_labels == 2:   
    criterion_dynamic = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 15]).to(DEVICE))
else:
    criterion_dynamic = nn.BCEWithLogitsLoss()
    

parameters_to_train = []
for key in models.keys():
    parameters_to_train += list(models[key].parameters())

model_optimizer = torch.optim.Adam(parameters_to_train,
                            lr=opt.learning_rate,
                            weight_decay=1e-5)

model_lr_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, 
            opt.scheduler_step_size, 0.1)

#########################################################################################

## Training Loop
learning_curve_st = []
learning_curve_dy = []

print('model initialized')
for epoch in range(num_epochs):
    for key in models.keys():
        models[key].to(DEVICE)
        models[key].train()
    train_static_ts = []
    train_static_loss = 0
    #-----------------
    train_dynamic_loss = 0
    train_dynamic_ts = []
    #----------------
    total_train_loss = 0
    sample_size = 0
    start_time = time.time()
    batch_end_time = start_time

    for batch, (sample, target, road_image, extra) in enumerate(train_loader):
        batch_size = len(sample)

        single_cam_inputs = []
        for i in range(num_images):
            single_cam_input = torch.stack([batch[i] for batch in sample])
            single_cam_input = Variable(single_cam_input).to(DEVICE)
            single_cam_inputs.append(single_cam_input)

        # ===================forward=====================
        encoded_features = models['encode'](single_cam_inputs)
        # print('encoder shapes', encoded_features.shape) -> torch.Size([9, 64, 96, 64])
        outputs= {}
        outputs['static'] = models['static'](encoded_features)
        outputs['dynamic'] = models['dynamic'](encoded_features)

        # true 
        road_image_long = torch.stack(road_image).type(torch.FloatTensor).to(DEVICE) # torch.Size([9, 800, 800]) 
        bbox_matrix_long = torch.stack([utils.bounding_box_to_matrix_image(i, labels = labels) for i in target]).to(DEVICE) # torch.Size([9, 800, 800])
        if opt.verbose:
            print('static & dynamic ground truth shape', road_image_long.shape, bbox_matrix_long.shape)
            print('static & dynamic prediction shape & type', outputs['static'].shape, outputs['dynamic'].shape,
                 outputs['static'].dtype, outputs['dynamic'].dtype)
        # loss
        if opt.num_dynamic_labels == 10:  
            loss_dynamic = criterion_dynamic(outputs['dynamic'].to(DEVICE), bbox_matrix_long)
        elif opt.num_dynamic_labels == 2:
            loss_dynamic = criterion_dynamic(outputs["dynamic"].to(DEVICE), bbox_matrix_long.to(DEVICE))
            #print(loss_dynamic)
        else:
            loss_dynamic = criterion_dynamic(outputs["dynamic"].view(-1,800,800).to(DEVICE), bbox_matrix_long.type(torch.float).to(DEVICE))
        loss_static = criterion_static(outputs['static'].view(-1,800,800).to(DEVICE), road_image_long.to(DEVICE))
        
        total_loss = loss_dynamic + loss_static
        # ===================backward====================
        model_optimizer.zero_grad()
        total_loss.backward()
        model_optimizer.step()
        # ===================log========================
        train_dynamic_loss += loss_dynamic.item() * batch_size
        train_static_loss += loss_static.item() * batch_size
        total_train_loss += total_loss.item() + train_static_loss
        sample_size += batch_size
        # static
        batch_ts, _ = utils.get_ts_for_batch_binary(outputs["static"], road_image)
        train_static_ts.extend(batch_ts)
        avg_train_ts = sum(batch_ts) / len(batch_ts)
        # dynamic
        current_batch_dynamic_ts, _ = utils.get_ts_for_bb(outputs["dynamic"], target, opt.num_dynamic_labels)
        train_dynamic_ts.extend(current_batch_dynamic_ts)
        avg_dynamic_ts_batch = sum(current_batch_dynamic_ts) / batch_size
        # log
        batch_time = time.time() - batch_end_time
        batch_end_time = time.time()
        if batch % 10 == 0:
            print('batch [{}], epoch [{}], loss: {:.4f}, \
                  time: {:.0f}s, static_ts: {:.4f}, dynamic_ts: {:.4f}'
                  .format(batch + 1, epoch + 1, total_train_loss / sample_size,
                          batch_time, avg_train_ts, avg_dynamic_ts_batch))
            if opt.verbose:
                print(f'dynamic loss: {train_dynamic_loss/sample_size}, \
                        static loss: {train_static_loss/sample_size}')
        # Empty Cache
        torch.cuda.empty_cache()

    # update lr
    model_lr_scheduler.step()

    # ===================log every epoch======================
    train_ts_static = sum(train_static_ts) / len(train_static_ts)
    if len(train_dynamic_ts) == 0: 
        train_ts_dynamic = 0
    else:
        train_ts_dynamic = sum(train_dynamic_ts) / len(train_dynamic_ts)
    val_ts_static, predicted_val_map_st, val_ts_dynamic, predicted_val_map_dy = utils.evaluation_unet(models, val_loader, DEVICE, opt.num_dynamic_labels)
    time_this_epoch_min = (time.time() - start_time) / 60
    print('epoch [{}/{}], loss: {:.4f}, time: {:.2f}min, remaining: {:.2f}min, \
        train_static_ts: {:.4f}, val_static_ts: {:.4f}, \
        train_dynamic_ts: {:.4f}, val_dynamic_ts: {:.4f}'
          .format(epoch + 1, num_epochs, total_train_loss / sample_size,
                  time_this_epoch_min, time_this_epoch_min * (num_epochs - epoch - 1), 
                  train_ts_static, val_ts_static, train_ts_dynamic, val_ts_dynamic,
                   ))

    learning_curve_st.append((train_ts_static.item(), val_ts_static.item()))
    learning_curve_dy.append((train_ts_dynamic.item(), val_ts_dynamic.item()))
    if epoch % 5 == 0 and epoch != 0:
        torch.save({
            'encode_state_dict': models['encode'].state_dict(),
            'decode_static_state_dict': models['static'].state_dict(),
            'decode_dynamic_state_dict': models['dynamic'].state_dict(),
            'plot_cache': [learning_curve_st, learning_curve_dy],
            'predicted_val_static_map': predicted_val_map_st,
            'predicted_val_dynamic_map': predicted_val_map_dy
        },
            FOLDER_PATH + 'hexa_{}_{}.pth'.format(timestampStr, epoch))

torch.save({
    'encode_state_dict': models['encode'].state_dict(),
    'decode_dynamic_state_dict': models['dynamic'].state_dict(),
    'decode_static_state_dict': models['static'].state_dict(),
    'plot_cache': [learning_curve_st, learning_curve_dy],
    'predicted_val_static_map': predicted_val_map_st,
    'predicted_val_dynamic_map': predicted_val_map_dy
},
    FOLDER_PATH + 'hexa_{}_final.pth'.format(timestampStr))
