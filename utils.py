from tqdm import tqdm

import numpy as np
import PIL.Image as pil
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

import networks # Source from monodepth2
from layers import disp_to_depth # Source from monodepth2
from torch.autograd import Variable
import skimage.measure
# import cv2

import code.helper as helper
import code.data_helper as data_helper

num_images = data_helper.NUM_IMAGE_PER_SAMPLE


class RandomBatchSampler(torch.utils.data.Sampler):
    """Sample random batches with sequential data samples.

    When getting inputs of [0, 1, 2, 3, 4, 5, 6, 7, 8] with batch_size=2,
    returns [4, 5], [0, 1], [8], [2, 3], [6, 7] in a random sequence.
    """
    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        all_batches = []
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                all_batches.append(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            all_batches.append(batch)

        rand_index = torch.randperm(len(all_batches)).tolist()
        for index in rand_index:
            yield all_batches[index]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def evaluation(model, data_loader, device):
    """
    Evaluate the model using thread score.

    Args:
        model: A trained pytorch model.
        data_loader: The dataloader for a labeled dataset.

    Returns:
        Average threat score for the entire data set.
        Predicted classification results.
    """
    model.eval()
    model.to(device)
    ts_list = []
    predicted_maps = []
    with torch.no_grad():
        for sample, _, road_image, extra in tqdm(data_loader):
            single_cam_inputs = []
            for i in range(num_images):
                single_cam_input = torch.stack([batch[i] for batch in sample])
                single_cam_input = Variable(single_cam_input).to(device)
                single_cam_inputs.append(single_cam_input)

            bev_output = model(single_cam_inputs)
            batch_ts, predicted_road_map = get_ts_for_batch_binary(bev_output, road_image)
            ts_list.extend(batch_ts)
            predicted_maps.append(predicted_road_map)
    return np.nanmean(ts_list), predicted_maps

def evaluation_layout(models, encoder_model_list, depth_decoder_model_list, data_loader, device):
    """
    Evaluate the model using thread score.
    Args:
        model: A trained pytorch model.
        data_loader: The dataloader for a labeled dataset.
    Returns:
        Average threat score for the entire data set.
        Predicted classification results.
    """
    depth_width = 320
    depth_height = 256
    original_width = 306
    original_height = 256
    transform = torchvision.transforms.ToTensor()
    models = to_eval(models, device)
    ts_list = []
    val_bscore_list = []
    predicted_maps = []
    with torch.no_grad():
        for sample, target, road_image, extra in tqdm(data_loader):
            # target_bb_map = torch.stack([bounding_box_to_matrix_image(i) for i in target]).to(device)
            single_cam_inputs = []
            raw_depth_inputs = []
            total_ats_bounding_boxes = 0
            
            for i in range(num_images):
                single_cam_input = torch.stack([batch[i] for batch in sample])
                # Need to get depth outputs as well, since the required input size for depth model is 320 * 256,
                # need to do some processing
                input_for_depth_list = []
                for j in range(len(sample)):
                    im = single_cam_input[j]
                    pil_im = torchvision.transforms.ToPILImage()(im)
                    pil_im = pil_im.resize((depth_width, depth_height), pil.LANCZOS)
                    input_for_depth_list.append(transform(pil_im))
                input_for_depth = torch.stack(input_for_depth_list).to(device)
                # Get outputs of depth model
                with torch.no_grad():
                    d_features = encoder_model_list[i](input_for_depth)
                    disp_outputs = depth_decoder_model_list[i](d_features)[("disp", 0)]
                # We need to change the outputs dimension back to original dimension
                disp_resized = nn.functional.interpolate(disp_outputs, (original_height, original_width), mode="bilinear", align_corners=False)
                # Now we need to transfer the output disparity map to depth map
                _, depth_out = disp_to_depth(disp_resized, 1, 40) # We set min depth to be 1 and max depth be 40
                # Reproduce the only channel 3 times for it to match with encoder
                depth_out = torch.cat([depth_out, depth_out, depth_out], 1)
            
                raw_depth_inputs.append(depth_out)
                single_cam_input = Variable(single_cam_input).to(device)
                single_cam_inputs.append(single_cam_input)

            encoded_raw_features = models['raw_encoder'](single_cam_inputs)
            encoded_depth_features = models['depth_encoder'](raw_depth_inputs)
        
            # Now we concat the features from raw input and depth input
            encoded_features = torch.cat([encoded_raw_features, encoded_depth_features], 1)
            outputs = {}
            outputs["dynamic"] = models["dynamic_decoder"](encoded_features)
            outputs["static"] = models["static_decoder"](encoded_features)

            # Add for bbox part
            predicted_bb_map = (outputs["dynamic"].view(-1, 800, 800) > 0)
        
            # Get bbox prediction from bb_maps
            bbox_list = []
            for i in range(len(sample)):
                bbox_list.append(image_to_bbox(predicted_bb_map[i]))
                
            # Calculate Bounding Box Score
            for i in range(len(sample)):
                try:
                    ats_bounding_boxes = helper.compute_ats_bounding_boxes(bbox_list[i], target[i]['bounding_box'])
                except:
                    ats_bounding_boxes = 0
                total_ats_bounding_boxes += ats_bounding_boxes
            val_bscore_list.append(total_ats_bounding_boxes / len(sample))

            batch_ts, predicted_road_map = get_ts_for_batch_binary(outputs["static"], road_image)
            ts_list.extend(batch_ts)
            predicted_maps.append(predicted_road_map)
    return np.nanmean(ts_list),np.nanmean(val_bscore_list), predicted_maps


def to_train(models, DEVICE):
    for key in models.keys():
        models[key].to(DEVICE)
        models[key].train()
    return models


def to_eval(models, DEVICE):
    for key in models.keys():
        models[key].to(DEVICE)
        models[key].eval()
    return models


def get_ts_for_batch(model_output, road_image):
    """Get average threat score for a mini-batch.

    Args:
        model_output: A matrix as the output from the classification model with a shape of
            (batch_size, num_classes, height, width).
        road_image: A matrix as the truth for the batch with a shape of
            (batch_size, height, width).

    Returns:
        Average threat score.
    """
    _, predicted_road_map = model_output.max(1)
    predicted_road_map = predicted_road_map.type(torch.BoolTensor)
    # predicted_road_map = np.argmax(bev_output.cpu().detach().numpy(), axis=1).astype(bool)

    batch_ts = []
    for batch_index in range(len(road_image)):
        sample_ts = helper.compute_ts_road_map(predicted_road_map[batch_index].cpu(),
                                               road_image[batch_index])
        batch_ts.append(sample_ts)
    return batch_ts, predicted_road_map


def get_ts_for_batch_binary(model_output, road_image):
    """Get average threat score for a mini-batch.

    Args:
        model_output: A matrix as the output from the classification model with a shape of
            (batch_size, num_classes, height, width).
        road_image: A matrix as the truth for the batch with a shape of
            (batch_size, height, width).

    Returns:
        Average threat score.
    """
#     _, predicted_road_map = model_output.max(1)
#     predicted_road_map = predicted_road_map.type(torch.BoolTensor)
    predicted_road_map = (model_output > 0.5).view(-1, 800, 800)
    # predicted_road_map = np.argmax(bev_output.cpu().detach().numpy(), axis=1).astype(bool)

    batch_ts = []
    for batch_index in range(len(road_image)):
        sample_ts = helper.compute_ts_road_map(predicted_road_map[batch_index].cpu(),
                                               road_image[batch_index])
        batch_ts.append(sample_ts)
    return batch_ts, predicted_road_map


def combine_six_to_one(samples):
    """Combine six samples or feature maps into one.
        [sample0][sample1][sample2]
        [sample3][sample4][sample5], with the second row in vertically flipped direction.

    Can also try combining them along features.

    Args:
        samples: TODO

    Returns: TODO

    """
    return torch.rot90(
        torch.cat(
            [torch.cat(samples[:3], dim=-1),
             torch.cat([torch.flip(i, dims=(-2, -1)) for i in samples[3:]], dim=-1)
            ], dim=-2), k=3, dims=(-2, -1))


def bounding_box_to_matrix_image(one_target, labels=True):
    """Turn bounding box coordinates and labels to 800x800 matrix with label on the corresponding index.
    Args:
        one_target: target[i] TODO
    Returns: TODO
    """
    bounding_box_map = np.zeros((800, 800))

    for idx, bb in enumerate(one_target['bounding_box']):
        label = one_target['category'][idx]
        min_y, min_x = np.floor((bb * 10 + 400).numpy().min(axis=1))
        max_y, max_x = np.ceil((bb * 10 + 400).numpy().max(axis=1))
        # print(min_x, max_x, min_y, max_y)
        for i in range(int(min_x), int(max_x)):
            for j in range(int(min_y), int(max_y)):
                if labels:
                    bounding_box_map[-i][j] = label + 1
                else:
                    bounding_box_map[-i][j] = 1
    return torch.from_numpy(bounding_box_map).type(torch.LongTensor)

# Write a function to transfer prediction to bounding boxes

def image_to_bbox(image):
    labe = skimage.measure.label(image.cpu())
    region_proposals = skimage.measure.regionprops(labe)
    num_bbox = len(region_proposals)
    bboxes = np.zeros([num_bbox, 2, 4])
    
    for i, rp in enumerate(region_proposals):
        raw_bb = np.array(rp.bbox)
        raw_bb[-2:] = raw_bb[-2:] - 1 # Since bbox in rp is upperbound exclusive
        y_min, x_min, y_max, x_max = (raw_bb - 400) / 10
        bbox = np.array([x_min, x_min, x_max, x_max, -y_min, -y_max, -y_min, -y_max]).reshape(2,4)
        bboxes[i] = bbox
    
    return torch.from_numpy(bboxes)

