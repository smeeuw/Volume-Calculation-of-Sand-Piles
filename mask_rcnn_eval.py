import json
import os
from typing import List, Union, Tuple

import torch
import copy
from cjm_pytorch_utils.core import move_data_to_device
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.detection import intersection_over_union
from torchmetrics.functional.classification import binary_precision
from torchmetrics.detection import IntersectionOverUnion

from stockpile_dataset import StockpileDataset
from utils import unpack_stockpile_dataset_elements, select_n_random, \
    get_model_prediction_as_dict, load_mask_rcnn_on_device, convert_uint8_tensor_to_bool_tensor


##########################################################################################################
# All functions in this file are only intended for BINARY instance segmentation tasks. If you have
# a multi-class problem, you should probably calculate mAP with an official library. Currently no GPU
# support. Batch processing could need refactoring as well.
##########################################################################################################

def calculate_in_batches(batch_size, function, **kwargs):
    images = None
    masks = None
    bboxes = None
    mode = None
    device = None
    model = None
    for key, value in kwargs.items():
        if key == 'images':
            images = value
        elif key == 'masks':
            masks = value
        elif key == 'boxes':
            bboxes = value
        elif key == 'mode':
            mode = value
        elif key == 'device':
            device = value
        elif key == 'model':
            model = value
    inputs = [model, images, bboxes, masks, device, mode]

    total_images = images.shape[0]
    assert total_images > 0, "There must be a positive number of images"
    result_float = 0.0
    result_float2 = 0.0
    output_is_tuple = False
    for start in range(0, total_images, batch_size):
        input_copy = copy.deepcopy(inputs)
        end = min(start + batch_size, total_images)
        # create batches from tensors
        for idx, current in enumerate(input_copy):
            if isinstance(current, torch.Tensor) or isinstance(current, list):
                input_copy[idx] = input_copy[idx][start:end]
        result = function(*[arg for arg in input_copy if arg is not None])

        if isinstance(result, float):
            result_float = result_float + (result * (end - start))
        elif isinstance(result, Tuple):
            output_is_tuple = True
            result1 = result[0]
            result2 = result[1]
            result_float = result_float + (result1 * (end - start))
            result_float2 = result_float2 + (result2 * (end - start))

    if output_is_tuple:
        return result_float / total_images, result_float2 / total_images
    else:
        return result_float / total_images


def find_avg_box_iou(
        model: nn.Module,
        images_gt: torch.Tensor,
        bbox_gts: List[torch.Tensor],
        device: torch.device,
        threshold: float = 0.5
) -> float:
    """
    Find the average Intersection over Union (IoU) between ground truth bounding boxes and model predictions.
    The bounding boxes should be in 'xyxy' format. If no matching was possible for the entire batch, 0 will be returned.

    Args:
        model(nn.Module): The model that performs the predictions.
        images_gt (torch.Tensor): Image tensor. Should be [N, C, H, W] torch.uint8 or torch.float32.
            N ist the number of images.
        bbox_gts: List of ground truth bounding boxes. Each entry is [N, 4] torch.float32, where N is the number
            of detected bounding boxes on the image.
        device (torch.device): Device for inference.
        threshold (float): IoU threshold for considering a match between predicted and ground truth bounding boxes.
            Only IoU values of matched boxes above this threshold will be considered for the final calculation.
            Defaults to 0.5.
    Returns:
        float: Average IoU value for the entire [N, C, H ,W] image batch.

    Raises:
        ValueError: If the input images are not of the shape [N, C, H, W].

    Notes:
        - The function handles both CPU and GPU devices. If the specified device is not CPU, it moves
          the data and model predictions to the GPU.
    """
    model.eval()

    if not isinstance(images_gt, torch.Tensor):
        raise ValueError("Image must be [N, C, H, W] tensor")

    if device.type != 'cpu':
        # Move to GPU if it is not already on gpu
        images_gt_gpu = move_data_to_device(images_gt, device)
        bbox_gts_gpu = move_data_to_device(bbox_gts, device)
        model_predictions = get_model_prediction_as_dict(model, images_gt_gpu, 0.5,
                                                         device=device)  # find model predictions
        computed_iou = compute_average_box_iou(bbox_gts_gpu, model_predictions['bboxes'], threshold)
        model.train()  # safety
        return computed_iou

    model_predictions = get_model_prediction_as_dict(model, images_gt, 0.5)  # find model predictions
    computed_iou = compute_average_box_iou(bbox_gts, model_predictions['bboxes'])
    model.train()  # safety
    return computed_iou


def compute_average_box_iou(bboxes_gts: List[torch.Tensor], bboxes_model: List[torch.Tensor],
                            threshold: float = 0.5) -> float:
    """
    Compute the average Intersection over Union (IoU) between ground truth bounding boxes and model predictions.
    The grounding boxes should be in 'xyxy' format.
    For each ground truth box, the best predicted box will be taken for the calculation. IoU values of matched boxes
    below the threshold will be discarded (not considered for the calculation) because they are considered
    to not be the same box. There are more sophisticated matching approaches like Hungarian Matching,
    but they take more computational resources. If no matching was possible in the entire batch, 0 will be returned.

    Args:
        bboxes_gts (List[torch.Tensor]): List of ground truth bounding box tensors for each image. Each element
            in the list is a [N, 4] torch.float32 Tensor, where N is the number of ground truth bounding boxes.
        bboxes_model (List[torch.Tensor]): List of model predicted bounding box tensors for each image. Each element
            in the list is a [N, 4] torch.float32 Tensor, where N is the number of detected bounding boxes.
        threshold (float): IoU threshold for considering a match between predicted and ground truth bounding boxes.
            Only IoU values of matched boxes above this threshold will be considered for the final calculation.
            Defaults to 0.5.

    Returns:
        float: Average IoU across all images.
    """
    average_iou = 0
    calculation_counter = 0

    for bounding_boxes_gt, bbox_model in zip(bboxes_gts, bboxes_model):
        # cannot find IoU in this case, since gt or predictions have no boxes
        if bounding_boxes_gt.numel() == 0 or bbox_model.numel() == 0:
            continue

        # gives back iou matrix
        metric = intersection_over_union(preds=bbox_model,
                                         target=bounding_boxes_gt, aggregate=False)

        # find best matches for each gt box along column
        best_match_boxes = torch.max(metric, dim=0)[0]

        # filter values above threshold
        above_threshold = best_match_boxes[best_match_boxes > threshold]

        if above_threshold.numel() == 0:  # cannot calculate IoU if no possible matching
            continue

        calculation_counter += 1
        average_iou += (above_threshold.sum().item() / above_threshold.numel())

    if calculation_counter == 0:  # no possible matching in entire batch will just return 0
        return 0

    return average_iou / calculation_counter


def find_avg_mask_iou(
        model: nn.Module,
        images_gt: torch.Tensor,
        masks_gts: List[torch.Tensor],
        device: torch.device,
        threshold: float = 0.5
) -> float:
    """
    Compute the average Intersection over Union (IoU) between ground truth masks and model predictions.
    If no matching was possible for the entire batch, 0 will be returned.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        images_gt (torch.Tensor): Image tensor. Should be [N, C, H, W] torch.uint8 or torch.float32.
            N ist the number of images.
        masks_gts (List[torch.Tensor]): The ground truth masks corresponding to the images. Each element
            in the list should be a torch.uint8 tensor (with entries that are either 0 or 1)
            or a torch.bool Tensor.
        device (torch.device): The device to run the computation on ('cpu' or 'cuda').
        threshold (float, optional): IoU threshold for considering a match between predicted and ground truth masks.
            Only IoU values of matched masks above this threshold will be considered for the final calculation.
            Defaults to 0.5.

    Returns:
        float: The average IoU for the mask predictions.

    Raises:
        ValueError: If `images_gt` is not a torch.Tensor.

    Notes:
        - The function handles both CPU and GPU devices. If the specified device is not CPU, it moves
          the data and model predictions to the GPU.
    """
    model.eval()

    if not isinstance(images_gt, torch.Tensor):
        raise ValueError("Image must be [N, C, H, W] tensor")

    if device.type != 'cpu':
        # Move to GPU if it is not already on gpu
        images_gt_gpu = move_data_to_device(images_gt, device)
        masks_gts_gpu = move_data_to_device(masks_gts, device)
        model_predictions = get_model_prediction_as_dict(model, images_gt_gpu, 0.5,
                                                         device=device)  # find model predictions
        computed_iou = compute_average_mask_iou(masks_gts_gpu, model_predictions['masks'], threshold)
        model.train()  # safety
        return computed_iou

    model_predictions = get_model_prediction_as_dict(model, images_gt, 0.5)  # find model predictions
    computed_iou = compute_average_mask_iou(masks_gts, model_predictions['masks'])
    model.train()  # safety
    return computed_iou


def compute_average_mask_iou(masks_gts: List[torch.Tensor], masks_model: List[torch.Tensor],
                             iou_threshold: float = 0.5) -> float:
    """
    Compute the average Intersection over Union (IoU) between ground truth masks and model predictions.
    For each ground truth mask, the best predicted mask will be taken for the calculation. IoU values of matched masks
    below the threshold will be discarded (not considered for the calculation) because they are considered
    to not be the same mask. If no matching was possible in the entire batch, 0 will be returned.

    Args:
        masks_gts (List[torch.Tensor]): List of ground truth mask tensors for each image. Each element
            in the list should be a torch.uint8 tensor (with entries that are either 0 or 1)
            or a torch.bool Tensor.
        masks_model (List[torch.Tensor]): List of model predicted mask tensors for each image. Each element
            in the list should be a torch.uint8 tensor (with entries that are either 0 or 1)
            or a torch.bool Tensor.
        iou_threshold (float): IoU threshold for considering a match between predicted and ground truth masks.
            Only IoU values of matched masks above this threshold will be considered for the final calculation.
            Defaults to 0.5.

    Returns:
        float: Average IoU across all images.
    """
    average_iou = 0
    calculation_counter = 0
    for masks_gt, masks_pred in zip(masks_gts, masks_model):
        if masks_gt.numel() == 0 or masks_pred.numel() == 0:
            # cannot calculate IoU with empty masks
            continue

        # calculate IoU matrix - find best predicted box for each gt box
        best_iou_values = []
        for mask_gt in masks_gt:
            best_iou = 0.0
            for mask_pred in masks_pred:
                mask_gt_bin = convert_uint8_tensor_to_bool_tensor(mask_gt)
                mask_pred_bin = convert_uint8_tensor_to_bool_tensor(mask_pred)
                iou = compute_binary_mask_iou(mask_gt_bin, mask_pred_bin)
                if iou > best_iou:
                    best_iou = iou
            best_iou_values.append(best_iou)

        best_iou_tensor = torch.tensor(best_iou_values, device=masks_gt.device)

        # filter values above threshold
        above_threshold = best_iou_tensor[best_iou_tensor > iou_threshold]

        if above_threshold.numel() == 0:  # cannot calculate IoU if no possible matching
            continue

        calculation_counter += 1
        average_iou += (above_threshold.sum().item() / above_threshold.numel())

    if calculation_counter == 0:  # no possible matching in entire batch will just return 0
        return 0

    return average_iou / calculation_counter


def compute_binary_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    Args:
        mask1 (torch.Tensor): torch.bool tensor representing the first mask.
        mask2 (torch.Tensor): torch.bool tensor representing the second mask.

    Returns:
        float: The IoU (Intersection over Union) value between the two masks.
    """
    intersection = (mask1 * mask2).sum().item()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum().item()
    return intersection / union


def find_avg_precision_and_recall(
        model: nn.Module,
        images_gt: torch.Tensor,
        bbox_or_mask_gts: List[torch.Tensor],
        device: torch.device,
        mode: str,
        threshold: float = 0.5,
) -> Tuple[float, float]:
    """
        Compute the average precision and average recall for bounding boxes or masks depending on `mode`.

        Args:
            model (nn.Module): The PyTorch model to evaluate.
            images_gt (torch.Tensor): Image tensor. Should be [N, C, H, W] torch.uint8 or torch.float32.
                N ist the number of images.
            bbox_or_mask_gts (List[torch.Tensor]): List of ground truth bounding box/mask tensors for each image.
                Each element in the list is a tensor representing torch.float32 bounding boxes in [N, 4] 'xyxy' format
                for mode='box' or torch.unit8 tensors (with entries of either 0 or 1) or torch.bool tensors for
                mode='mask'.
            device (torch.device): The device to run the computation on ('cpu' or 'cuda').
            mode (str): Mode of operation. Must be either 'box' or 'mask' indicating whether bounding boxes or masks
                are being used.
            threshold (float, optional): Only IoU values of matched masks (boxes) above this threshold will be considered
                as True Positive. Defaults to 0.5.

        Returns:
            Tuple[float, float]: Average precision and average recall.

        Raises:
            ValueError: If `images_gt` is not a torch.Tensor.
            AssertionError: If `mode` is not 'box' or 'mask'.

        Notes:
            - The function handles both CPU and GPU devices. If the specified device is not CPU, it moves
              the data and model predictions to the GPU.
        """
    model.eval()

    if not isinstance(images_gt, torch.Tensor):
        raise ValueError("Image must be [N, C, H, W] tensor")

    assert mode in ["mask", "box"], "Mode must be mask or box"

    if device.type != 'cpu':
        # Move to GPU if it is not already on gpu
        images_gt_gpu = move_data_to_device(images_gt, device)
        bbox_or_mask_gts_gpu = move_data_to_device(bbox_or_mask_gts, device)
        model_predictions = get_model_prediction_as_dict(model, images_gt_gpu, 0.5,
                                                         device=device)  # find model predictions

        if mode == 'box':
            avg_precision, avg_recall = compute_average_precision_and_average_recall(bbox_or_mask_gts_gpu,
                                                                                     model_predictions['bboxes'],
                                                                                     mode, threshold)
        else:
            avg_precision, avg_recall = compute_average_precision_and_average_recall(bbox_or_mask_gts_gpu,
                                                                                     model_predictions['masks'],
                                                                                     mode, threshold)

        model.train()  # safety
        return avg_precision, avg_recall

    model_predictions = get_model_prediction_as_dict(model, images_gt, 0.5)  # find model predictions

    if mode == 'box':
        avg_precision, avg_recall = compute_average_precision_and_average_recall(bbox_or_mask_gts,
                                                                                 model_predictions['bboxes'],
                                                                                 mode, threshold)
    else:
        avg_precision, avg_recall = compute_average_precision_and_average_recall(bbox_or_mask_gts,
                                                                                 model_predictions['masks'],
                                                                                 mode, threshold)
    model.train()  # safety
    return avg_precision, avg_recall


def compute_average_precision_and_average_recall(bboxes_or_masks_gts: List[torch.Tensor],
                                                 bboxes_or_masks_model: List[torch.Tensor],
                                                 mode: str, iou_threshold: float = 0.5) -> Tuple[float, float]:
    """
        Compute the average precision and average recall between ground truth bounding boxes/masks and model predictions.
        With `mode` it can be specified, if the calculation should be done for 'box' or 'mask'. When matching
        masks (boxes) the predicted mask (box) will be matched with the best fitting ground truth mask (box).
        Best fit is calculated based on IoU.

        Args:
            bboxes_or_masks_gts (List[torch.Tensor]): List of ground truth bounding box/mask tensors for each image.
                Each element in the list is a tensor representing torch.float32 bounding boxes in [N, 4] 'xyxy' format
                for mode='box' or torch.unit8 tensors (with entries of either 0 or 1) or torch.bool tensors for
                mode='mask'.
            bboxes_or_masks_model (List[torch.Tensor]): List of model predicted bounding box/mask tensors for each image.
                 Each element in the list is a tensor representing torch.float32 bounding boxes in [N, 4] 'xyxy' format
                 for mode='box' or torch.unit8 tensors (with entries of either 0 or 1) or torch.bool tensors for
                 mode='mask'.
            mode (str): Mode of operation. Must be either 'box' or 'mask' indicating whether bounding boxes or masks
                are being used.
            iou_threshold (float): IoU threshold for considering a match between predicted and ground truth masks
            (boxes). Only IoU values of matched masks (boxes) above this threshold will be considered
            as True Positive. Defaults to 0.5.

        Returns:
            Tuple[float, float]: Average precision and average recall across all images.

        Raises:
            AssertionError: If mode is not 'box' or 'mask'.

        Notes:
            - The function has to address several edge cases. Most notably, when a given ground truth mask (box) is the
              best fit for multiple predicted boxes, the first predicted box will be a True Positive and every other box
              will be a False Positive.
            - When there are no ground truth bounding boxes/masks and no model predicted bounding boxes/masks, metrics
              cannot be computed. If this happens for every image, 0 will be returned.
            - When there are no ground truth bounding boxes/masks but there are model predicted bounding boxes/masks,
              precision is 0 because every prediction is a false positive (FP),
              and recall is meaningless due to the absence of ground truth.
            - When there are ground truth bounding boxes/masks but no model predicted bounding boxes/masks,
              recall is 0 because no ground truth box/mask is predicted.
              Precision is meaningless as there are no predictions to evaluate against ground truth.
        """

    assert mode in ["mask", "box"], "Mode must be mask or box"

    average_precision = 0
    average_recall = 0
    number_images_precision = len(bboxes_or_masks_gts)
    number_images_recall = len(bboxes_or_masks_gts)

    for bounding_boxes_or_masks_gt, bbox_or_masks_model in zip(bboxes_or_masks_gts, bboxes_or_masks_model):
        # edge cases - if no polygon is on the image, gt_boxes will be empty tensor
        # if no prediction was made, model_boxes will be empty tensor
        if bounding_boxes_or_masks_gt.numel() == 0 and bbox_or_masks_model.numel() == 0:
            number_images_precision -= 1
            number_images_recall -= 1
            continue  # do nothing, since metric is not calculated in these cases
        elif bounding_boxes_or_masks_gt.numel() == 0 and bbox_or_masks_model.numel() != 0:
            # every prediction is a fp so precision is 0, recall is meaningless since no gt boxes
            number_images_recall -= 1
            continue
        elif bounding_boxes_or_masks_gt.numel() != 0 and bbox_or_masks_model.numel() == 0:
            # no gt box is predicted leading to recall of zero, since no predictions precision is meaningless
            number_images_precision -= 1
            continue

        # normal cases
        tp_plus_fn = bounding_boxes_or_masks_gt.shape[0]  # number of bboxes on image
        tp_plus_fp = bbox_or_masks_model.shape[0]  # number of predicted bboxes
        tp_tensor = torch.zeros(tp_plus_fp, dtype=torch.uint8, device=bbox_or_masks_model.device)
        fp_tensor = torch.zeros(tp_plus_fp, dtype=torch.uint8, device=bbox_or_masks_model.device)
        matched_gt_boxes_or_masks = torch.zeros(tp_plus_fn, dtype=torch.uint8, device=bbox_or_masks_model.device)
        for pred_idx, predicted_boxes_or_masks in enumerate(bbox_or_masks_model):
            best_iou = 0.0
            best_gt_idx = -1
            predicted_boxes_or_masks = predicted_boxes_or_masks.unsqueeze(0)
            for gt_idx, gt_boxes_or_masks in enumerate(bounding_boxes_or_masks_gt):
                gt_boxes_or_masks = gt_boxes_or_masks.unsqueeze(0)
                if mode == 'box':
                    iou = intersection_over_union(preds=predicted_boxes_or_masks, target=gt_boxes_or_masks)
                else:
                    mask_gt_bin = convert_uint8_tensor_to_bool_tensor(gt_boxes_or_masks)
                    mask_pred_bin = convert_uint8_tensor_to_bool_tensor(predicted_boxes_or_masks)
                    iou = compute_binary_mask_iou(mask_gt_bin, mask_pred_bin)
                if iou > best_iou:  # find best match for predicted box
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx == -1:  # no overlap at all between the predicted box and any gt box
                fp_tensor[pred_idx] = 1
            else:
                if best_iou > iou_threshold:
                    if matched_gt_boxes_or_masks[best_gt_idx] == 0:  # predicted box was not matched to gt box before
                        tp_tensor[pred_idx] = 1
                        matched_gt_boxes_or_masks[best_gt_idx] = 1
                    else:  # predicted box was matched to gt box before
                        fp_tensor[pred_idx] = 1
                else:  # box does not match with any gt box above IoU threshold
                    fp_tensor[pred_idx] = 1

        matched_boxes_or_masks_sum = matched_gt_boxes_or_masks.sum().item()
        tp_count = tp_tensor.sum().item()
        fp_count = fp_tensor.sum().item()
        fn_count = matched_gt_boxes_or_masks.shape[0] - matched_boxes_or_masks_sum
        # division by zero cannot happen, is covered in edge cases
        average_precision += tp_count / (tp_count + fp_count)
        average_recall += tp_count / (tp_count + fn_count)

    if number_images_precision == 0:
        number_images_precision = 1e-7

    if number_images_recall == 0:
        number_images_recall = 1e-7

    return average_precision / number_images_precision, average_recall / number_images_recall


def compute_f1_score(precision: float, recall: float) -> float:
    """
    Compute the F1 score from precision and recall values.

    Args:
        precision (float): The precision value.
        recall (float): The recall value.

    Returns:
        float: The F1 score calculated from the precision and recall.

    """
    return (2 * precision * recall) / (precision + recall + 1e-7)


if "__main__" == __name__:
    from visualise import display_maskrcnn_predictions_and_ground_truth

    BATCH_SIZE = 50

    path_folder = "mask_rcnn_models/trained_model_rgb_blend_1024_1024_2_resnet_b4_after_refactor"
    trainable_layers = 2
    number_of_classes = 2
    mode = 'rgb_blend'

    train_dataset = (StockpileDataset.create_builder(dataset_mode='test', image_mode=mode)
                     .set_has_been_built(False)).build()

    model = load_mask_rcnn_on_device(os.path.join(path_folder, "mask_rcnn_resnet50_fpn.pt"), torch.device('cpu'),
                                     trainable_layers=trainable_layers, mode=mode, dataset=train_dataset,
                                     number_of_classes=number_of_classes)

    elements, masks, bboxes, labels, image_indices = unpack_stockpile_dataset_elements(
        select_n_random(train_dataset, len(train_dataset)), mode=mode)

    avg_mask_iou = calculate_in_batches(BATCH_SIZE, find_avg_mask_iou, model=model, images=elements, masks=masks,
                                        device=torch.device('cpu'))

    print(f"avg mask iou: {avg_mask_iou}")

    avg_box_iou = calculate_in_batches(BATCH_SIZE, find_avg_box_iou, model=model, images=elements, boxes=bboxes,
                                       device=torch.device('cpu'))

    print(f"avg box iou: {avg_box_iou}")

    avg_precision_box, average_recall_box = calculate_in_batches(BATCH_SIZE, find_avg_precision_and_recall, model=model,
                                                                 images=elements,
                                                                 boxes=bboxes, device=torch.device('cpu'), mode='box')

    print(f"avg precision: {avg_precision_box}")
    print(f"avg recall: {average_recall_box}")

    avg_precision_mask, average_recall_mask = calculate_in_batches(BATCH_SIZE, find_avg_precision_and_recall,
                                                                   model=model, images=elements,
                                                                   masks=masks, device=torch.device('cpu'),
                                                                   mode='mask')
    print(f"avg precision: {avg_precision_mask}")
    print(f"avg recall: {average_recall_mask}")

    f1_score_box = compute_f1_score(avg_precision_box, average_recall_box)
    f1_score_mask = compute_f1_score(avg_precision_mask, average_recall_mask)

    print(f"f1 score box: {f1_score_box}")
    print(f"f1 score mask: {f1_score_mask}")

    results_dict = {
        'avg_mask_iou': avg_mask_iou,
        'avg_box_iou': avg_box_iou,
        'avg_precision_box': avg_precision_box,
        'avg_precision_mask': avg_precision_mask,
        'avg_recall_box': average_recall_box,
        'avg_recall_mask': average_recall_mask,
        'f1_score_box': f1_score_box,
        'f1_score_mask': f1_score_mask
    }

    with open(os.path.join(path_folder, "results_test"), 'w') as file:
        json.dump(results_dict, file)

    # print(find_avg_mask_iou(model, elements, masks, device=torch.device('cpu')))
    # print(find_avg_mask_iou(model, elements, bboxes, device=torch.device('cpu')))
    # print(find_avg_precision_and_recall(model, elements, masks, mode='mask', device=torch.device('cpu')))
    # print(find_avg_precision_and_recall(model, elements, bboxes, mode='box', device=torch.device('cpu')))

    # display_maskrcnn_predictions_and_ground_truth(elements, masks, bboxes, labels, train_dataset.label_map,
    #                                               path_model="mask_rcnn_models/trained_model_rgb_blend_512_512/mask_rcnn_resnet50_fpn.pt")

    # model.eval()
    #
    # display_maskrcnn_predictions_and_ground_truth(elements, masks, bboxes, labels, train_dataset.label_map, "mask_rcnn_models/trained_model_e_350_1024_1024/mask_rcnn_resnet50_fpn.pt")
    #
    # model_predictions = get_model_prediction_as_dict(model, elements, 0.5)
    # model_mask = model_predictions['masks']
    # model_bboxes = model_predictions['bboxes']
    # model_labels = model_predictions['labels']
    #
    # model_res = [{'boxes': model_bboxes[0], 'labels': model_labels[0]}]
    # gt_res = [{'boxes': bboxes[0], 'labels': labels[0]}]
    #
    # metric = IntersectionOverUnion(iou_threshold=0.5)
    # print(metric(model_res, gt_res))

    # data_loader_params = {
    #     'batch_size': TrainConfig.BATCH_SIZE,  # Batch size for data loading
    #     'num_workers': TrainConfig.NUMBER_WORKERS,  # Number of subprocesses to use for data loading
    #     'persistent_workers': True,
    #     # If True, the data loader will not shutdown the worker processes after a dataset has been consumed once.
    #     # This allows to maintain the worker dataset instances alive.
    #     'pin_memory': 'cuda' in TrainConfig.DEVICE,
    #     # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using
    #     # GPU.
    #     'pin_memory_device': TrainConfig.DEVICE if 'cuda' in TrainConfig.DEVICE else '',
    #     # Specifies the device where the data should be loaded. Commonly set to use the GPU.
    #     'collate_fn': lambda batch: tuple(zip(*batch)),
    # }
    #
    # valid_dataloader = DataLoader(train_dataset, **data_loader_params)
