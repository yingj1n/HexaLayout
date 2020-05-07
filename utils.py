from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable
# import cv2

import code.helper as helper
import code.data_helper as data_helper
import skimage.measure

from shapely.geometry import Polygon

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


def evaluation(models, data_loader, device, dynamic_label): ## changed to add bbox matrix prediction
    """
    Evaluate the model using thread score.

    Args:
        model: A trained pytorch model.
        data_loader: The dataloader for a labeled dataset.

    Returns:
        Average threat score for the entire data set.
        Predicted classification results.
    """
    for key in models.keys():
        models[key].to(device)
    ts_list = []
    predicted_static_maps = []
    predicted_dynamic_maps = []
    #iou_list = []
    dynamic_ts_list = []
    with torch.no_grad():
        for sample, target, road_image, extra in tqdm(data_loader):
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
            # bbox_matrix= torch.tensor(bounding_box_to_matrix_image(target[0], dynamic_label)).to(device) 
            # _, output_dynamic_pred = torch.max(outputs['dynamic'], dim = 1)
            # mean_iou = compute_bbox_matrix_iou(output_dynamic_pred, bbox_matrix)
            bb_batch_ts, predicted_bb_map = get_ts_for_bb(outputs["dynamic"], target, dynamic_label)
            # static
            batch_ts, predicted_road_map = get_ts_for_batch_binary(outputs['static'], road_image)
           
            # log 
            ts_list.extend(batch_ts)
            predicted_static_maps.append(predicted_road_map)
            predicted_dynamic_maps.append(predicted_bb_map)
            #iou_list.append(mean_iou)
            dynamic_ts_list.extend(bb_batch_ts)
        
    return np.nanmean(ts_list), predicted_static_maps, np.nanmean(dynamic_ts_list), predicted_dynamic_maps


def get_ts_for_bb(model_output, target, bbox_labels):
    if bbox_labels == 10:
        _, predicted_bb_map = model_output.max(1)
        predicted_bb_map = predicted_bb_map.type(torch.BoolTensor)
    else:
        predicted_bb_map = (model_output > 0.5).view(-1, 800, 800)

    batch_ts = []
    for batch_index in range(len(target)):
        predicted_boxes = matrix_to_bbox(predicted_bb_map[batch_index].cpu())
        sample_ts = compute_ats_bounding_boxes(predicted_boxes,
                                               target[batch_index]['bounding_box'])
        batch_ts.append(sample_ts)
    return batch_ts, predicted_bb_map


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

    predicted_road_map = (model_output > 0.5).view(-1, 800, 800)
 
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


def bounding_box_to_matrix_image(one_target, num_labels=10):
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
                if num_labels == 10:
                    bounding_box_map[-i][j] = label + 1
                else:
                    bounding_box_map[-i][j] = 1
    return torch.from_numpy(bounding_box_map).type(torch.LongTensor)

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

def compute_bbox_matrix_iou(batch_predictions, batch_labels, n_classes = 10):
    '''
    given two matrices of true labels and predictions, return the mean iou over 10 classes
    TODO: change this to avg mean threat scores (to get bbox from matrix)
    '''
    num_batches = batch_predictions.shape[0]
    mean_iou_list = []
    for i in range(num_batches):
        labels, predictions = batch_labels[i], batch_predictions[i]

        mean_iou = 0.0
        seen_classes = 0
        for c in range(n_classes):
            labels_c = (labels != c)
            pred_c = (predictions != c)

            labels_c_sum = (labels_c).sum()
            pred_c_sum = (pred_c).sum()

            if (labels_c_sum > 0) or (pred_c_sum > 0):
                seen_classes += 1

                intersect = np.logical_and(labels_c.cpu(), pred_c.cpu()).sum()
                union = labels_c_sum + pred_c_sum - intersect

                mean_iou += intersect / union
        mean_iou = mean_iou / seen_classes if seen_classes else 0
        mean_iou_list.append(mean_iou)

    return np.nanmean(mean_iou_list)


def compute_ats_bounding_boxes(boxes1, boxes2):
    num_boxes1 = boxes1.size(0)
    num_boxes2 = boxes2.size(0)

    boxes1_max_x = boxes1[:, 0].max(dim=1)[0]
    boxes1_min_x = boxes1[:, 0].min(dim=1)[0]
    boxes1_max_y = boxes1[:, 1].max(dim=1)[0]
    boxes1_min_y = boxes1[:, 1].min(dim=1)[0]

    boxes2_max_x = boxes2[:, 0].max(dim=1)[0]
    boxes2_min_x = boxes2[:, 0].min(dim=1)[0]
    boxes2_max_y = boxes2[:, 1].max(dim=1)[0]
    boxes2_min_y = boxes2[:, 1].min(dim=1)[0]

    condition1_matrix = (boxes1_max_x.unsqueeze(1) > boxes2_min_x.unsqueeze(0))
    condition2_matrix = (boxes1_min_x.unsqueeze(1) < boxes2_max_x.unsqueeze(0))
    condition3_matrix = (boxes1_max_y.unsqueeze(1) > boxes2_min_y.unsqueeze(0))
    condition4_matrix = (boxes1_min_y.unsqueeze(1) < boxes2_max_y.unsqueeze(0))
    condition_matrix = condition1_matrix * condition2_matrix * condition3_matrix * condition4_matrix

    iou_matrix = torch.zeros(num_boxes1, num_boxes2)
    for i in range(num_boxes1):
        for j in range(num_boxes2):
            if condition_matrix[i][j]:
                iou_matrix[i][j] = compute_iou(boxes1[i], boxes2[j])

    iou_max = iou_matrix.max(dim=0)[0]

    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    total_threat_score = 0
    total_weight = 0
    for threshold in iou_thresholds:
        tp = (iou_max > threshold).sum()
        threat_score = tp * 1.0 / (num_boxes1 + num_boxes2 - tp)
        total_threat_score += 1.0 / threshold * threat_score
        total_weight += 1.0 / threshold

    average_threat_score = total_threat_score / total_weight
    
    return average_threat_score

def compute_iou(box1, box2):
    a = Polygon(torch.t(box1)).convex_hull
    b = Polygon(torch.t(box2)).convex_hull
    
    return a.intersection(b).area / a.union(b).area

def matrix_2d_to_3d(matrix, num_classes = 2):
    ''' 
    Use this function to tweek grond truth matrices to a three dimensional matrices,
    from (H * W) to (num_classes * H * W), which in each dimension it is hot encoded features
    :param matrix: a 2d np array
    :num_classes : number of classes in this matrix
    :output: return a 3d np array that one hot encoded each category
    '''
    x, y = matrix.shape
    output = np.zeros((num_classes, x, y))
    for i in range(num_classes):
        for xx in range(x):
            for yy in range(y):
                if matrix[xx][yy] == i:
                    output[i, xx, yy] = 1
    return torch.from_numpy(output).type(torch.LongTensor)

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
