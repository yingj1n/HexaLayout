from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable
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
