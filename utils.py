from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import skimage.measure
# from dl_final_project.layers import disp_to_depth  # Source from monodepth2
# import cv2

import PIL.Image as pil

import code.helper as helper
import code.data_helper as data_helper

num_images = data_helper.NUM_IMAGE_PER_SAMPLE

import sklearn.metrics as metrics


def get_accuracy_auc_for_batch(outputs, targets):
    accuracy = []
    aucs = []
    m = nn.Softmax(dim=1)
    output_pos_prob = m(outputs)
    _, predicted_map = outputs.max(1)
    predicted_map = predicted_map.type(torch.BoolTensor)

    for i, target in enumerate(targets):
        
        acc = metrics.accuracy_score(target.flatten().detach().cpu(),
                                     predicted_map[i].flatten().detach().cpu())
        accuracy.append(acc)
        fpr, tpr, thresholds = metrics.roc_curve(
            target.flatten().detach().cpu(),
            output_pos_prob[i][1].flatten().detach().cpu(), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)
    return accuracy, aucs


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


def evaluation_unet(models, data_loader, device, dynamic_label): ## changed to add bbox matrix prediction
    """
    Evaluate the model using thread score.
    Args:
        model: A trained pytorch model.
        data_loader: The dataloader for a labeled dataset.
    Returns:
        verage threat score for the entire data set.
        Predicted classification results.
    """
    for key in models.keys():
        models[key].to(device)
        models[key].eval()
    ts_list = []
    predicted_static_maps = []
    predicted_dynamic_maps = []
    #iou_list = []
    dynamic_ts_list = []
    with torch.no_grad():
        for i, (sample, target, road_image, extra) in enumerate(data_loader):
            if i % 50 == 0:
                print('-'*10,'evaluation at sample', i,'-'*10)
            single_cam_inputs = []
            for i in range(num_images):
                single_cam_input = torch.stack([batch[i] for batch in sample])
                single_cam_input = Variable(single_cam_input).to(device)
                single_cam_inputs.append(single_cam_input)
            
            encoded_features = models['encode'](single_cam_inputs)
            outputs= {}
            outputs['static'] = models['static'](encoded_features)
            outputs['dynamic'] = models['dynamic'](encoded_features)

            # dynamic
            bb_batch_ts, predicted_bb_map = get_ts_for_bb(outputs["dynamic"], target, dynamic_label)
            # static
            batch_ts, predicted_road_map = get_ts_for_batch_binary(outputs['static'], road_image)
           
            # log 
            ts_list.extend(batch_ts)
            predicted_static_maps.append(predicted_road_map)
            predicted_dynamic_maps.append(predicted_bb_map)
            dynamic_ts_list.extend(bb_batch_ts)
        
    return np.nanmean(ts_list), predicted_static_maps, np.nanmean(dynamic_ts_list), predicted_dynamic_maps


def evaluation(model, data_loader, device):
    """
    Evaluate the roadmap model using thread score.

    Args:
        model: A trained pytorch model.
        data_loader: The dataloader for a labeled dataset.
        device: the torch.device to evaluation on

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
            batch_ts, predicted_road_map = get_rm_ts_for_batch_binary(bev_output, road_image)
            ts_list.extend(batch_ts)
            predicted_maps.append(predicted_road_map)
    return np.nanmean(ts_list), predicted_maps


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


def evaluation_layout(models, data_loader, device, bbox_labels=False,
                      depth=False, encoder_model_list=None, depth_decoder_model_list=None):
    """
    Evaluate the layout model using thread score.

    Args:
        model: A trained pytorch model.
        data_loader: The dataloader for a labeled dataset.
        device: the torch.device to evaluation on
        bbox_labels: whether to use 10 bbox labels (as opposed to treating all as 1
        depth: whether to add depth in input
        encoder_model_list, depth_decoder_model_list: only applicable if depth = True

    Returns:
        Average threat score for the entire data set.
        Predicted classification results.
    """
    models = to_eval(models, device)
    rm_ts_list = []
    bb_ts_list = []
    rm_acc_list = []
    rm_auc_list = []
    bb_acc_list = []
    bb_auc_list = []
    predicted_maps = []
    trans_normalize = torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                       std=(0.5, 0.5, 0.5))
    with torch.no_grad():
        for sample, target, road_image, extra in tqdm(data_loader):
            # target_bb_map = torch.stack([bounding_box_to_matrix_image(i) for i in target]).to(device)
            target_bb_map = torch.stack([bounding_box_to_matrix_image(i, True) for i in target]).to(device)
            single_cam_inputs = []
            for i in range(num_images):
                single_cam_input = torch.stack([batch[i] for batch in sample])
                if depth:
                    single_cam_input = torch.stack([batch[i] for batch in sample])
                    depth_out = get_predicted_depth(encoder_model_list[i],
                                                    depth_decoder_model_list[i],
                                                    single_cam_input, device)
                    single_cam_input = torch.stack([trans_normalize(batch) for batch in single_cam_input])
                    single_cam_input = Variable(torch.cat((single_cam_input.to(device), depth_out), 1))
                else:
#                     single_cam_input = torch.stack([trans_normalize(batch) for batch in single_cam_input])
                    single_cam_input = Variable(single_cam_input).to(device)
                single_cam_inputs.append(single_cam_input)

            encoded_features = models['encoder'](single_cam_inputs)
            outputs = {}
            outputs["dynamic"] = models["dynamic_decoder"](encoded_features)
            outputs["static"] = models["static_decoder"](encoded_features)

            roadmap_batch_ts, predicted_road_map = get_rm_ts_for_batch(outputs["static"], road_image)
            rm_batch_accu, rm_batch_auc = get_accuracy_auc_for_batch(outputs["static"], road_image)
            bb_batch_ts, predicted_bb_map = get_bb_ts_for_batch(outputs["dynamic"], target)
            bb_batch_accu, bb_batch_auc = get_accuracy_auc_for_batch(outputs["dynamic"], target_bb_map)
            #             print(roadmap_batch_ts, bb_batch_ts)
            rm_ts_list.extend(roadmap_batch_ts)
            rm_acc_list.extend(rm_batch_accu)
            rm_auc_list.extend(rm_batch_auc)
            bb_ts_list.extend(bb_batch_ts)
            bb_acc_list.extend(bb_batch_accu)
            bb_auc_list.extend(bb_batch_auc)
            predicted_maps.append(predicted_road_map)  # useless
            
        accu_auc = {'rm': [np.nanmean(rm_acc_list), np.nanmean(rm_auc_list)],
                    'bb': [np.nanmean(bb_acc_list), np.nanmean(bb_auc_list)]}

    return np.nanmean(rm_ts_list), np.nanmean(bb_ts_list), accu_auc


def get_predicted_depth(encoder_model, decoder_model, single_sam_samples, device):
    depth_width = 320
    depth_height = 256
    original_width = 306
    original_height = 256
    transform = torchvision.transforms.ToTensor()
    input_for_depth_list = []
    for j in range(len(single_sam_samples)):
        im = single_sam_samples[j]
        pil_im = torchvision.transforms.ToPILImage()(im)
        # the required input size for depth model is 320 * 256
        pil_im = pil_im.resize((depth_width, depth_height), pil.LANCZOS)
        input_for_depth_list.append(transform(pil_im))
    input_for_depth = torch.stack(input_for_depth_list).to(device)
    # Get outputs of depth model
    with torch.no_grad():
        d_features = encoder_model(input_for_depth)
        disp_outputs = decoder_model(d_features)[("disp", 0)]
    # We need to change the outputs dimension back to original dimension
    disp_resized = nn.functional.interpolate(disp_outputs, (original_height, original_width), mode="bilinear",
                                             align_corners=False)
    # Now we need to transfer the output disparity map to depth map
    _, depth_out = disp_to_depth(disp_resized, 1, 40)  # We set min depth to be 1 and max depth be 40
    # Reproduce the only channel 3 times for it to match with encoder
    return depth_out


def get_bb_ts_for_batch(model_output, target):
    _, predicted_bb_map = model_output.max(1)
    predicted_bb_map = predicted_bb_map.type(torch.BoolTensor)

    batch_ts = []
    for batch_index in range(len(target)):
        predicted_boxes = image_to_bbox(predicted_bb_map[batch_index].cpu())
        sample_ts = helper.compute_ats_bounding_boxes(predicted_boxes,
                                                      target[batch_index]['bounding_box'])
        batch_ts.append(sample_ts)
    return batch_ts, predicted_bb_map


def get_rm_ts_for_batch(model_output, road_image):
    """Get average roadmap threat score for a mini-batch.

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


def get_rm_ts_for_batch_binary(model_output, road_image):
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


def bounding_box_to_matrix_image(one_target, labels=True, outter=True):
    """Turn bounding box coordinates and labels to 800x800 matrix with label on the corresponding index.

    Args:
        one_target: target[i] TODO

    Returns: TODO

    """
    bounding_box_map = np.zeros((800, 800))

    for idx, bb in enumerate(one_target['bounding_box']):
        label = one_target['category'][idx]
        if outter:
            min_y, min_x = np.ceil((bb * 10 + 400).numpy().min(axis=1))
            max_y, max_x = np.floor((bb * 10 + 400).numpy().max(axis=1))
        else:
            min_y, min_x = (bb * 10 + 400).numpy().min(axis=1)
            max_y, max_x = (bb * 10 + 400).numpy().max(axis=1)
            others_y = [i * 10 + 400 for i in bb[0].numpy() if i * 10 + 400 not in (min_y, max_y)]
            others_x = [i * 10 + 400 for i in bb[1].numpy() if i * 10 + 400 not in (min_x, max_x)]
            min_y, max_y = np.floor(min(others_y)), np.ceil(max(others_y))
            min_x, max_x = np.floor(min(others_x)), np.ceil(max(others_x))
        # print(min_x, max_x, min_y, max_y)
        for i in range(int(min_x), int(max_x)):
            for j in range(int(min_y), int(max_y)):
                if labels:
                    bounding_box_map[-i][j] = label + 1
                else:
                    bounding_box_map[-i][j] = 1
    return torch.from_numpy(bounding_box_map).type(torch.LongTensor)


def image_to_bbox(image):
    labe = skimage.measure.label(image)
    region_proposals = skimage.measure.regionprops(labe)
    num_bbox = len(region_proposals)
    bboxes = np.zeros([num_bbox, 2, 4])

    for i, rp in enumerate(region_proposals):
        raw_bb = np.array(rp.bbox)
        raw_bb[-2:] = raw_bb[-2:] - 1  # Since bbox in rp is upperbound exclusive
        y_min, x_min, y_max, x_max = (raw_bb - 400) / 10
        bbox = np.array([x_min, x_min, x_max, x_max, -y_min, -y_max, -y_min, -y_max]).reshape(2, 4)
        bboxes[i] = bbox

    if bboxes.any():
        return torch.from_numpy(bboxes)
    else:  # Return the entire canvas when no bbox is being identified.
        bboxes = np.zeros([1, 2, 4])
        bboxes[0] = np.array([0, 0, 800, 800, -0, -800, -0, -800]).reshape(2, 4)
        return torch.from_numpy(bboxes)



def bounding_box_to_3d_matrix_image(one_target, num_labels=10):
    """Turn bounding box coordinates and labels to 800x800 matrix with label on the corresponding index.
    Args:
        one_target: target[i] TODO
    Returns: TODO
    """
    bounding_box_map = np.zeros((num_labels, 800, 800))
    bounding_box_map[0] = 1 # mark all background into 1
    for idx, bb in enumerate(one_target['bounding_box']):
        label = one_target['category'][idx]
        min_y, min_x = np.floor((bb * 10 + 400).numpy().min(axis=1))
        max_y, max_x = np.ceil((bb * 10 + 400).numpy().max(axis=1))
        # print(min_x, max_x, min_y, max_y)
        for i in range(int(min_x), int(max_x)):
            for j in range(int(min_y), int(max_y)):
                bounding_box_map[label+1][-i][j] = 1
                bounding_box_map[0][-i][j] = 0
    return torch.from_numpy(bounding_box_map).type(torch.LongTensor)

def road_map_to_3d_matrix(matrix):
    '''
    takes in a batch of matrices and return a 4d matrices by adding a dimension and one hot encoding road/non-road
    :input: matrix, a dim of batch * H * W matrices with 0's and 1's
    :output: matrix_3d, a dim of batch * 2 * H * W matrices one hot encoded road/non-road matrices. 
    '''
    # print('input matrix shape', matrix.shape)
    # print('matrix[0]', matrix[0].shape, type(matrix[0]))
    batch, x, y = matrix.shape
    matrix_3d = torch.empty(batch, 2, x, y)
    for i in range(batch):
        matrix_3d[i, :, :, :] = torch.stack((matrix[i], 1 - matrix[i]), 0)
    return matrix_3d

def matrix_to_3d_matrix(matrix):
    '''
    takes in a batch of matrices and return a 4d matrices by adding a dimension and one hot encoding road/non-road
    :input: matrix, a dim of batch * H * W matrices with 0's and 1's
    :output: matrix_3d, a dim of batch * 2 * H * W matrices one hot encoded road/non-road matrices. 
    '''
    batch, H, W = matrix.shape
    matrix_3d = torch.zeros((batch,1,H,W))
    matrix_3d = matrix[:, None,:,:]
    return matrix_3d


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

    predicted_road_map = (model_output > 0.5).view(-1, 800, 800)
 
    batch_ts = []
    for batch_index in range(len(road_image)):
        sample_ts = helper.compute_ts_road_map(predicted_road_map[batch_index].cpu(),
                                               road_image[batch_index])
        batch_ts.append(sample_ts)
    return batch_ts, predicted_road_map

def get_ts_for_bb(model_output, target, bbox_labels):
    if bbox_labels == 10:
        _, predicted_bb_map = model_output.max(1)
        predicted_bb_map = predicted_bb_map.type(torch.BoolTensor)
    else:
        predicted_bb_map = (model_output > 0.5).view(-1, 800, 800)

    batch_ts = []
    for batch_index in range(len(target)):
        predicted_boxes = matrix_to_bbox(predicted_bb_map[batch_index].cpu())
        sample_ts = helper.compute_ats_bounding_boxes(predicted_boxes,
                                               target[batch_index]['bounding_box'])
        batch_ts.append(sample_ts)
    return batch_ts, predicted_bb_map


def matrix_to_bbox(image, verbose = False):
    image = image.cpu()
    label = skimage.measure.label(image)
    region_proposals = skimage.measure.regionprops(label)
    num_bbox = len(region_proposals)
    if verbose:
        print('input image shape', image.shape)
        print('partial input image', image[:3,:3])
        print('number of bbox to return: ', num_bbox)
    bboxes = np.zeros([num_bbox, 2, 4])
    
    for i, rp in enumerate(region_proposals):
        raw_bb = np.array(rp.bbox)
        raw_bb[-2:] = raw_bb[-2:] - 1 # Since bbox in rp is upperbound exclusive
        y_min, x_min, y_max, x_max = (raw_bb - 400) / 10
        bbox = np.array([x_min, x_min, x_max, x_max, -y_min, -y_max, -y_min, -y_max]).reshape(2,4)
        bboxes[i] = bbox
    
    if num_bbox:
        return torch.from_numpy(bboxes)
    else: # Return the entire canvas when no bbox is being identified.
        bboxes = np.zeros([1, 2, 4])
        bboxes[0] = np.array([0, 0, 800, 800, -0, -800, -0, -800]).reshape(2,4)
        return torch.from_numpy(bboxes)

# Some functions used to project 6 images and combine into one.
# Requires cv2. Not currently used in modeling.

# def perspective_transform(image):
#     height, width, _ = image.shape
#     rect = np.array([
#         [0, height//2],
#         [width - 1, height//2],
#         [width - 1, height-1],
#         [0, height-1]], dtype = "float32")
#     dst = np.array([
#         [-180, -200],
#         [width + 180, -200],
#         [width - 130, height - 1],
#         [130, height-1]], dtype = "float32")
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (width, height))
#     return warped
#
# def image_transform_via_cv2(torch_image, angle):
#     numpy_image = torch_image.numpy().transpose(1, 2, 0)
#     perspective = perspective_transform(numpy_image)
#     rotation = rotate_image(perspective, angle)
#     numpy_transformed = torch.from_numpy(rotation)
#     torch_transformed = torch.transpose(torch.transpose(numpy_transformed, 0, 2), 1, 2)
#     return torch_transformed
#
# def rotate_image(image, angle):
#     image_center = tuple(np.array(image.shape[1::-1]) / 2)
#     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#     result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#     return result
#
# def images_transform(sample, idx):
#     angle = 60
#     rotation_angle = {0:angle, 1:0, 2:-angle, 3:-angle, 4:0, 5:angle}
#     post_rotation = []
#     for image in sample:
#         transformed = image_transform_via_cv2(image, rotation_angle[idx])
#         post_rotation.append(transformed)
#     return torch.stack(post_rotation)
