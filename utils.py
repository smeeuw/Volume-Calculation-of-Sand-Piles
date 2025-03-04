import json
import os
import sys
from itertools import zip_longest
from typing import List, Union, Tuple, Dict, Optional, Any

import numpy as np
import torch
from matplotlib.axes import Axes
from torch.amp import autocast
import torchvision
from albumentations import BaseCompose
from cjm_pytorch_utils.core import move_data_to_device
from distinctipy import distinctipy
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
import torch.nn.functional as F


def unpack_stockpile_dataset_elements(elements: List[Tuple[torch.Tensor, Dict[str, Any]]], mode: str = 'rgb') -> (
        Tuple)[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[int]]:
    """
        Unpacks elements from a dataset into separate lists for images, masks, bounding boxes, labels, and image IDs.

        Args:
            elements (List[Tuple[torch.Tensor, Dict[str, Any]]]): A list of tuples where each tuple contains an image
                tensor and a dictionary of annotations.
                The dictionary should have the keys 'masks', 'boxes', 'labels', and 'image_id'.
            mode (str, optional): A string indicating the mode of the image. Must be one of 'rgb',
                'rgb_blend', or 'rgbh'. Default is 'rgb'.

        Returns: A tuple of images, masks, bounding boxes, labels, and image IDs.
        Raises:
            AssertionError: If the mode is not one of 'rgb', 'rgb_blend', or 'rgbh'.
    """
    assert mode in ['rgb', 'rgb_blend', 'rgbh'], "Mode must be 'rgb' or 'rgbh' or 'rgb_blend'"
    elements_result, masks_result, bboxes_result, labels_result = [], [], [], []
    image_indices = []
    for image_tensor, tensor_dict in elements:
        elements_result.append(image_tensor)
        masks_result.append(tensor_dict['masks'])
        bboxes_result.append(tensor_dict['boxes'])
        labels_result.append(tensor_dict['labels'])
        image_indices.append(tensor_dict['image_id'])
    return torch.stack(elements_result), masks_result, bboxes_result, labels_result, image_indices


def select_n_random(dataset: Dataset, n: int):
    """
        Select n random elements from the given dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to select from.
            n (int): The number of elements to select.

        Returns:
            List: A list of n random elements from the dataset.
    """
    assert 0 < n <= len(dataset), ("Cannot select more elements than the dataset contains. Must select at least "
                                   "one.")
    shuffled_indices = torch.randperm(len(dataset))
    return [dataset[i] for i in shuffled_indices][:n]


def generate_n_distinct_colors(n, seed: Optional[int] = None) -> list[tuple[int, int, int]]:
    """
        Generates a list of n distinct RGB colors as tuples of integers.

        Args:
            n (int): The number of distinct colors to generate.
            seed (Optional[int]): Seed value for random number generation. Default is None.

        Returns:
            list[tuple[int, int, int]]: A list of tuples, where each tuple represents an RGB color
                as integers ranging from 0 to 255.
    """
    colors = distinctipy.get_colors(n, rng=seed)
    int_colors = [tuple(int(c * 255) for c in color) for color in colors]
    # noinspection PyTypeChecker
    return [tuple(sublist) for sublist in int_colors]


def validate_targets(elements: torch.Tensor, target: list[torch.Tensor]) -> bool:
    """
       Validates whether the target list matches the elements list in length and existence.

       Args:
           elements (torch.Tensor): The tensor against which to validate.
           target (list[torch.Tensor]): The target list to validate against `elements`. Should be of the same length as `elements`.

       Returns:
           bool: True if `target` is not None and its length matches `elements`, False otherwise.
       """
    return not (target is None or len(elements) != len(target))


def convert_hwc_np_array_to_chw_tensor(hwc_np_array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """
        Convert a NumPy array from HWC (height, width, channels) format to CHW (channels, height, width)
        format as a Torch tensor.

        Args:
            hwc_np_array (np.ndarray): Input NumPy array in HWC format to be converted.
            dtype (torch.dtype): Data type of the resulting Torch tensor.

        Returns:
            torch.Tensor: Converted Torch tensor in CHW format.
    """
    return torch.tensor(hwc_np_array.transpose(2, 0, 1), dtype=dtype)


def convert_bbox_list_to_n4_tensor(bbox: list[np.ndarray], dtype: torch.dtype) -> torch.Tensor:
    """
        Convert a list of NumPy arrays representing bounding boxes into a torch.Tensor.

        Args:
            bbox (list[np.ndarray]): A list where each element is a NumPy array of shape (4,) representing a bounding box.
            dtype (torch.dtype): The desired data type of the output tensor.

        Returns:
            A tensor of shape (N, 4), where N is the number of bounding boxes in `bbox`.
    """
    if len(bbox) == 0:
        return torch.empty((0, 4), dtype=dtype)
    else:
        return torch.tensor(bbox, dtype=dtype)


def convert_label_list_to_n_tensor(label_list: list[np.ndarray], dtype: torch.dtype) -> torch.Tensor:
    """
       Convert a list of NumPy arrays representing labels into a torch.Tensor.

       Args:
           label_list (list[np.ndarray]): A list where each element is a NumPy array representing a label.
           dtype (torch.dtype): The desired data type of the output tensor.

       Returns:
           torch.Tensor: A tensor of shape (N,), where N is the number of elements in `label_list`.
    """
    if len(label_list) == 0:
        return torch.empty((0,), dtype=dtype)
    else:
        return torch.tensor(label_list, dtype=dtype)


def create_mask_tensor(element: torch.Tensor, mask: torch.Tensor, mask_alpha: float, color_seed: int) -> torch.Tensor:
    """
       Create a tensor with segmentation masks overlaid on an input image tensor.

       Args:
           element (torch.Tensor): The input image tensor (of shape [C, H, W]) on which masks will be overlaid.
           mask (torch.Tensor): The segmentation masks tensor (of shape [N, H, W]) to overlay on the element.
               N should be the number of masks, and H, W should match the height and width of the element tensor.
           mask_alpha (float): The transparency level of the overlaid masks. Should be in the range [0, 1],
               where 0 is fully transparent and 1 is fully opaque.
           color_seed (int): The seed value for generating distinct colors for each mask.

       Returns:
           torch.Tensor: A tensor representing the input image element with segmentation masks overlaid.
               The tensor shape and type match the shape and type of the `element` input tensor.
    """
    annotated_tensor = draw_segmentation_masks(
        image=element,
        masks=mask,
        alpha=mask_alpha,
        colors=generate_n_distinct_colors(mask.shape[0], seed=color_seed)
    )
    return annotated_tensor


def create_bounding_box_tensor(element: torch.Tensor, bbox: torch.Tensor, labels: torch.Tensor,
                               label_map: Dict[int, str], color_seed: int,
                               bbox_width: int, bbox_font_size: int, bbox_font_file_path: str) -> torch.Tensor:
    """
       Create a tensor with bounding boxes and labels overlaid on an input image tensor.

       Args:
           element (torch.Tensor): The input image tensor (of shape [C, H, W]) on which bounding boxes and labels
               will be overlaid.
           bbox (torch.Tensor): The tensor of bounding box coordinates (of shape [N, 4]), where N is the number of
               bounding boxes. Each bounding box is represented as [x_min, y_min, x_max, y_max].
           labels (torch.Tensor): The tensor of labels corresponding to each bounding box (of shape [N]).
           label_map (Dict[int, str]): A dictionary mapping label indices to label names.
           color_seed (int): The seed value for generating distinct colors for each bounding box.
           bbox_width (int): The width of the bounding box lines in pixels.
           bbox_font_size (int): The font size of the label text inside the bounding box.
           bbox_font_file_path (str): The file path to the font file used
            for rendering label text inside bounding boxes.

       Returns:
           torch.Tensor: A tensor representing the input image element with bounding boxes and labels overlaid.
               The tensor shape and type match the shape and type of the `element` input tensor.
    """
    # no correct font file path specified
    if not os.path.exists(bbox_font_file_path):
        bbox_font_file_path = None

    labels_for_boxes = None
    if labels is not None:
        labels_for_boxes = [label_map[label.item()] for label in labels]

    annotated_tensor = draw_bounding_boxes(
        image=element,
        boxes=bbox,
        labels=labels_for_boxes,
        fill=False,
        colors=generate_n_distinct_colors(bbox.shape[0], seed=color_seed),
        width=bbox_width,
        font_size=bbox_font_size,
        font=bbox_font_file_path
    )
    return annotated_tensor


def apply_transformation_with_optional_workaround(element, transforms: BaseCompose, mask: Optional[torch.Tensor] = None,
                                                  bbox: Optional[torch.Tensor] = None,
                                                  label: Optional[torch.Tensor] = None,
                                                  use_workaround=False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
     Apply transformations to the input element, optionally using a workaround function if specified.

     Args:
         element (torch.Tensor): The input tensor or element to be transformed.
         transforms (BaseCompose): Composed transformations to be applied.
         mask (Optional[torch.Tensor], optional): Optional mask tensor for transformation.
         bbox (Optional[torch.Tensor], optional): Optional bounding box tensor for transformation.
         label (Optional[torch.Tensor], optional): Optional label tensor for transformation.
         use_workaround (bool, optional): Flag indicating whether to use a workaround function for transformations.

     Returns:
         torch.Tensor: The annotated tensor and a dictionary with the masks, bboxes and labels.

    """
    if use_workaround:
        return apply_transform_with_annotations_with_workaround(element.clone(), transforms, mask=mask, bbox=bbox,
                                                                label=label)
    else:
        return apply_transform_with_annotations(element.clone(), transforms, mask=mask, bbox=bbox, label=label)


def apply_transform_with_annotations(image: torch.Tensor, transform: BaseCompose, mask: torch.Tensor,
                                     bbox: torch.Tensor, label: torch.Tensor, transforms_seed: Optional[int] = None) \
        -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
        Apply a transformation to an image along with associated annotations (mask, bbox, label).

        Args:
            image (torch.Tensor): Input image tensor to be transformed.
            transform (BaseCompose): Albumentations Transformation to be applied.
            mask (torch.Tensor): Mask tensor associated with the image.
            bbox (torch.Tensor): Bounding box tensor associated with the image.
            label (torch.Tensor): Label tensor associated with the image.
            transforms_seed (Optional[int], optional): Random seed for deterministic transformations.

         Returns:
             torch.Tensor: The annotated tensor and a dictionary with the masks, bboxes and labels.
    """
    if transforms_seed is not None:
        torch.manual_seed(transforms_seed)
    transformed = transform(
        image=image.numpy(force=True).transpose((1, 2, 0)),
        mask=mask.numpy(force=True).transpose((1, 2, 0)),
        bboxes=bbox.numpy(force=True),
        bbox_classes=label.numpy(force=True)
    )

    image = transformed["image"]

    if not torch.is_tensor(image):
        image = convert_hwc_np_array_to_chw_tensor(image, torch.uint8)

    mask_transform = transformed["mask"]

    if not torch.is_tensor(mask_transform):
        mask_transform = convert_hwc_np_array_to_chw_tensor(mask_transform, torch.uint8)

    return image, {'masks': mask_transform,
                   'boxes': convert_bbox_list_to_n4_tensor(transformed["bboxes"]
                                                           , dtype=torch.float32),
                   'labels': convert_label_list_to_n_tensor(transformed["bbox_classes"]
                                                            , dtype=torch.int64)}


def apply_transform_with_annotations_with_workaround(
        image: torch.Tensor,
        transform: BaseCompose,
        mask: torch.Tensor,
        bbox: torch.Tensor,
        label: torch.Tensor,
        transforms_seed: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Apply a transformation to an image tensor with annotations, with a workaround for bugs in Albumentations.
    This workaround prevents two inconsistencies from Albumentations. 1) Albumentations returns empty masks,
    but not empty bounding boxes and labels, if they are no longer in the image.
    2) Albumentations cannot deal with masks of shape [0, H, W], which happens when no instance
    is on image. In this case, Albumentations still tries to apply the transformation,
    which leads to a segmentation fault.

    Args:
        image (torch.Tensor): Input image tensor. [C, H, W]
        transform (BaseCompose): Transformation function to apply.
        mask (torch.Tensor): Mask tensor. [N, H, W]
        bbox (torch.Tensor): Bounding box tensor. [N, 4]
        label (torch.Tensor): Label tensor. [N]
        transforms_seed (Optional[int]): Seed for reproducibility of transformations.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Tuple containing the transformed image tensor and annotation dictionary.

    Notes:
        - The function temporarily fills empty masks because transformations that interpolate cannot work with empty tensors in Albumentations.

        - After transformation, it ensures consistency between masks, bounding boxes, and labels, even if empty masks are removed.

        - Bounding boxes are regenerated from the transformed masks using torchvision's `masks_to_boxes` function.
    """
    if transforms_seed is not None:
        torch.manual_seed(transforms_seed)
    original_number_in_mask = mask.shape[0]
    # fill empty masks temporarily because Albumentations cannot work with empty tensors
    if mask.numel() == 0:
        mask = torch.zeros(1, mask.shape[1], mask.shape[2], dtype=torch.uint8)

    transformed = transform(
        image=image.numpy(force=True).transpose((1, 2, 0)),
        mask=mask.numpy(force=True).transpose((1, 2, 0)),
        bboxes=bbox.numpy(force=True),
        bbox_classes=label.numpy(force=True)
    )

    mask_transform = transformed["mask"]

    if not torch.is_tensor(mask_transform):
        mask_transform = convert_hwc_np_array_to_chw_tensor(mask_transform, torch.uint8)

    # copy label before transform to make masks and labels consistent
    # this is due to the fact that albumentations removes bounding boxes
    # if they degenerate, but they keep the empty masks
    labels = label
    if original_number_in_mask == 0:
        mask_transform = torch.empty((0, mask_transform.shape[1], mask_transform.shape[2]), dtype=torch.uint8)
    else:
        # check if mask is empty after operation and remove empty masks and corresponding labels
        for i in range(mask_transform.shape[0]):
            if torch.all(mask_transform[0] == 0):
                mask_transform = mask_transform[1:]
                labels = labels[1:]

    # regenerate bounding boxes
    bboxes_xyxy = torchvision.ops.masks_to_boxes(mask_transform)

    # now masks, bounding boxes, and labels are consistent

    image = transformed["image"]

    if not torch.is_tensor(image):
        image = convert_hwc_np_array_to_chw_tensor(image, torch.uint8)

    return image, {'masks': mask_transform, 'boxes': bboxes_xyxy, 'labels': labels}


def apply_transform_without_annotations(image: torch.Tensor, transform: BaseCompose,
                                        transforms_seed: Optional[int] = None) -> torch.Tensor:
    """
       Apply a transformation to an image without any associated annotations.

       Args:
           image (torch.Tensor): Input image tensor. [C, H, W]
           transform (BaseCompose): Transformation function to apply.
           transforms_seed (Optional[int]): Random seed for deterministic transformations.

       Returns:
           torch.Tensor: Transformed image tensor.
    """
    if transforms_seed is not None:
        torch.manual_seed(transforms_seed)
    transformed = transform(
        image=image.numpy().transpose((1, 2, 0)),
    )

    image = transformed["image"]

    if not torch.is_tensor(image):
        image = convert_hwc_np_array_to_chw_tensor(image, torch.uint8)

    return image


def convert_float32_tensor_to_uint8_tensor(to_transform: torch.Tensor) -> torch.Tensor:
    """
    Convert a float32 tensor in [0, 1] to a uint8 tensor by scaling values to [0, 255].

    Args:
        to_transform (torch.Tensor): Input tensor to be transformed.

    Returns:
        torch.Tensor: Transformed tensor with dtype torch.uint8. If the tensor is not of type torch.float32,
        the original tensor will be returned.
    """
    if to_transform.dtype == torch.float32:
        return (to_transform * 255.0).to(torch.uint8)
    else:
        return to_transform


def convert_uint8_tensor_to_float32_tensor(to_transform: torch.Tensor) -> torch.Tensor:
    """
    Convert a uint8 tensor to a float32 tensor by scaling values from [0, 255] to [0, 1].

    Args:
        to_transform (torch.Tensor): Input tensor to be transformed.

    Returns:
        torch.Tensor: Transformed tensor with dtype torch.float32. If the tensor is not of type torch.uint8,
        the original tensor will be returned.
    """
    if to_transform.dtype == torch.uint8:
        return to_transform.to(torch.float32) / 255.0
    else:
        return to_transform


def convert_uint8_tensor_to_bool_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a uint8 tensor to a boolean tensor. The uint8 tensors can only contain 0 or 1 as entry.

    Args:
        tensor (torch.Tensor): Input tensor to be transformed.

    Returns:
        torch.Tensor: Transformed boolean tensor. If the input tensor is not of type torch.uint8,
                      the original tensor will be returned.

    Raises:
        AssertionError: If the input tensor contains values other than 0 or 1.
    """
    if tensor.dtype == torch.uint8:
        assert tensor[tensor > 1].numel() == 0, "Tensor should only contain 0, 1, or combination of 0 and 1."
        return tensor.bool()
    else:
        return tensor


def load_mask_rcnn_on_device(path_model: str, device: torch.device, trainable_layers: int, number_of_classes: int,
                             mode: str = 'rgb', dataset=None) -> nn.Module:
    """
        Load a Mask R-CNN model on the specified device from a saved checkpoint.

        Args:
            path_model (str): Path to the saved model checkpoint.
            device (torch.device): Device to load the model onto (e.g., 'cpu' or 'cuda').
            trainable_layers (int): Number of trainable layers in the model's backbone.
            number_of_classes (int): Number of classes the model predicts.
            mode (str, optional): Mode of the model ('rgb', 'rgb_blend' or 'rgbh'). Defaults to 'rgb'.
            dataset (optional): Dataset object used for training or inference.

        Returns:
            torch.nn.Module: Loaded Mask R-CNN model.
    """
    from mask_rcnn_train_utils import initialise_maskrcnn
    from mask_rcnn_train_utils import initialise_maskrcnn_rgbh
    if mode == 'rgbh':
        model = initialise_maskrcnn_rgbh(device=device, trainable_backbone_layers=trainable_layers,
                                         dataset=dataset, number_of_classes=number_of_classes)
    else:
        model = initialise_maskrcnn(device=device, trainable_backbone_layers=trainable_layers,
                                    number_of_classes=number_of_classes)
    model.load_state_dict(torch.load(path_model))
    return model


def get_model_prediction_as_dict(model: nn.Module, images_gt: torch.Tensor, threshold: float,
                                 device=torch.device('cpu')) -> Dict[str, List[torch.Tensor]]:
    """
        Get predictions from a model as a dictionary containing bounding boxes, labels, scores, and masks.

        Args:
            model (torch.nn.Module): The model to perform predictions with.
            images_gt (torch.Tensor): Input images for prediction.
                Must be a torch.unit8 or torch.float32 [N, C, H, W] tensor.
            threshold (float): Threshold value for considering predicted instances.
            device (torch.device, optional): Device to perform prediction on. Defaults to torch.device('cpu').

        Returns:
           Dict[str, List[torch.Tensor]]

        Notes:
            - Keys in the returned dictionary are 'bboxes', 'labels', 'scores', 'masks'
            - 'bboxes': List of predicted bounding boxes. Each element in the list is a [N,4] torch.float32 tensor.
            - 'labels': List of predicted labels. Each element in the list is a [N] torch.int64 tensor.
            - 'scores': List of prediction scores. Each element in the list is a [N] torch.float32 tensor with probabilities.
            - 'masks': List of predicted masks. Each element is a torch.bool Tensor of shape [N, H, W].
            - N is the number of detected instances per entry in the list. The length of the returned list is the number of images.
        Raises:
            ValueError: If image_gt is not a tensor.
            AssertionError: If the number of images, masks, bounding boxes, labels, and prediction scores are not equal.
                If the provided tensor is not a [N, C, H, W] torch.uint8 or torch.float32 tensor.
    """
    if isinstance(images_gt, torch.Tensor):
        assert images_gt.shape[1] == 3 or images_gt.shape[1] == 1 or images_gt.shape[1] == 4, \
            "Images must be [N, C, H, W] tensor "
        assert images_gt.dtype == torch.float32 or images_gt.dtype == torch.uint8, \
            "Images must be torch.float32 or torch.uint8 Tensor"
        images_gt = convert_uint8_tensor_to_float32_tensor(images_gt)
        image_height = images_gt.shape[2]
        image_width = images_gt.shape[3]

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            images_gt = move_data_to_device(images_gt.detach(), device)

        model.eval()
        with torch.no_grad():
            if device.type == 'cuda':
                with autocast(device.type):
                    model_output = model(images_gt)
            else:
                model_output = model(images_gt)
    else:
        raise ValueError('Images must be torch.float32 or torch.uint8 [N, C, H, W] tensor')

    model_output = move_data_to_device(model_output, torch.device('cpu')) if device.type == 'cpu' else model_output

    scores_results = []
    mask_results = []
    bbox_results = []
    label_results = []

    for model_output in model_output:
        # only consider masks above threshold
        scores_mask = model_output['scores'] > threshold

        pred_bboxes = model_output['boxes'][scores_mask]
        pred_labels = model_output['labels'][scores_mask]
        pred_masks = model_output['masks'][scores_mask]
        pred_scores = model_output['scores'][scores_mask]

        result_masks = []
        for mask in pred_masks:
            if mask.numel() != 0:
                mask_tensor = torch.where(mask >= threshold, 1, 0).bool()
                result_masks.append(mask_tensor)

        if len(result_masks) == 0:
            result_mask = torch.empty((0, image_height, image_width), dtype=torch.bool)
        else:
            result_mask = torch.cat(result_masks)

        # append mask for each instance
        scores_results.append(pred_scores)
        label_results.append(pred_labels)
        bbox_results.append(pred_bboxes)
        mask_results.append(result_mask)

    assert len(images_gt) == len(scores_results) == len(label_results) == len(bbox_results) == len(mask_results), \
        "There must be an equal number of images, masks, bboxes, labels, and prediction scores"

    return {'bboxes': bbox_results, 'labels': label_results,
            'scores': scores_results, 'masks': mask_results}


def normalise_image_tensor_data_0_to_1(input_tensor: torch.Tensor, max_norm: Optional[float] = None,
                                       min_norm: Optional[float] = None,
                                       no_data_value: Optional[Union[float, int]] = None) -> torch.Tensor:
    """
        Normalize an input tensor to the range [0, 1]. If both, max_norm and min_norm are specified,
        the normalisation will be based on the specified values. Otherwise, the normalisation will be
        based on the maximum and minimum value of the tensor. The resulting tensor will be of type torch.float32.

        Args:
            input_tensor (torch.Tensor): Input tensor to be normalized.
            max_norm (Optional[float]): Maximum value for normalization.
            min_norm (Optional[float]): Minimum value for normalization.
            no_data_value (float or int or None): Value representing no data,
                which will be ignored during normalization.

        Returns:
            torch.Tensor: Normalized image tensor with a type of torch.float32.
    """
    result_tensor = torch.zeros(input_tensor.size(), dtype=torch.float32)

    if no_data_value is not None:
        valid_values_mask = find_valid_values_mask(input_tensor, no_data_value)
        valid_tensor = input_tensor[valid_values_mask]

        if valid_tensor.numel() == 0:
            # no valid values, so return tensor with only zeros
            return result_tensor
        else:
            if max_norm is not None and min_norm is not None:
                # global normalisation
                result_tensor[valid_values_mask] = normalise_tensor_0_to_1(valid_tensor, min_norm, max_norm)
                return result_tensor
            else:
                # local normalisation
                result_tensor[valid_values_mask] = normalise_tensor_0_to_1(valid_tensor, valid_tensor.min().item(),
                                                                           valid_tensor.max().item())
                return result_tensor
    else:
        if max_norm is not None and min_norm is not None:
            return normalise_tensor_0_to_1(input_tensor, min_norm, max_norm)
        else:
            return normalise_tensor_0_to_1(input_tensor, input_tensor.min().item(), input_tensor.max().item())


def find_valid_values_mask(tensor: torch.Tensor, no_data_value: Union[int, float]) -> torch.Tensor:
    """
    Create a mask to identify valid values based on a no_data_value.

    Args:
        tensor (torch.Tensor): Input tensor.
        no_data_value (float or int): Value representing no data.

    Returns:
        torch.Tensor: Boolean mask indicating valid values.
    """
    if isinstance(no_data_value, int):
        valid_values_mask = (tensor != no_data_value)
    elif isinstance(no_data_value, float):
        valid_values_mask = ~torch.isclose(tensor, torch.tensor(no_data_value))
    else:
        raise Exception('Invalid type for no_data_value. Must be either float or int.')

    return valid_values_mask


def normalise_tensor_0_to_1(tensor: torch.Tensor, tensor_min: float, tensor_max: float) -> torch.Tensor:
    """
        Normalize a tensor using given min and max values to the interval [0, 1].

        Args:
            tensor (torch.Tensor): Input tensor to be normalized.
            tensor_min (float): Minimum value for normalization.
            tensor_max (float): Maximum value for normalization.

        Returns:
            torch.Tensor: Normalized tensor.
        """
    epsilon = sys.float_info.epsilon
    result = torch.zeros(tensor.size(), dtype=torch.float32)
    if abs(tensor_max - tensor_min) < epsilon:
        return result
    else:
        result = (tensor - tensor_min) / (tensor_max - tensor_min)
        return result


def plot_comparison_images(axes: Axes, num_images: int, index: int, original_np: np.ndarray,
                           mod_np: np.ndarray, title_original: str, title_modified: str):
    """
        Plot comparison images side by side on specified matplotlib axes.

        Args:
            axes (matplotlib.axes.Axes): Axes object or array of Axes objects to plot the images on.
            num_images (int): Number of images to plot (1 or more).
            index (int): Index to place the images when `num_images > 1`.
            original_np (np.ndarray): Original image data as a numpy array.
            mod_np (np.ndarray): Modified image data as a numpy array.
            title_original (str): Title for the original image.
            title_modified (str): Title for the modified image.

        Returns:
            None
    """
    if num_images == 1:
        axes[0].imshow(original_np)
        axes[0].axis('off')
        axes[0].set_title(title_original)

        axes[1].imshow(mod_np)
        axes[1].axis('off')
        axes[1].set_title(title_modified)
    else:
        axes[index, 0].imshow(original_np)
        axes[index, 0].axis('off')
        axes[index, 0].set_title(title_original)

        axes[index, 1].imshow(mod_np)
        axes[index, 1].axis('off')
        axes[index, 1].set_title(title_modified)


def parse_training_results_json(json_file_path: str, target_key: str, train_or_val: str = None) -> Tuple[
    List[float], List[int]]:
    """
        Parse a JSON file containing training results and extract specific values based on `target_key`.

        Parameters:
        - json_file_path (str): Path to the JSON file containing training results.
        - target_key (str): Key indicating which value(s) to extract from the JSON data.
        - train_or_val (str, optional): Specifies whether to extract data from 'train' or 'val' subset.

        Returns: Extracted value(s) corresponding to `target_key` and the epochs of these values in a list.

        Raises:
        - AssertionError: If `train_or_val` is specified but is not 'train' or 'val'.
        - ValueError: If `json_file_path` does not exist or cannot be read,
            or if `target_key` is not found in the JSON data.
    """
    # read dict
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    assert train_or_val is None or train_or_val in ['train', 'val'], ("If specified, train_or_val should "
                                                                      "be either 'train' or 'val'")
    if isinstance(data, dict):
        return parse_single_dict_instance(data, target_key, train_or_val)
    else:
        result_data = []
        result_epochs = []
        for entry in data:
            parsed_result = parse_single_dict_instance(entry, target_key, train_or_val)
            result_data.append(parsed_result[0][0])
            result_epochs.append(parsed_result[1][0])
        return result_data, result_epochs


def parse_single_dict_instance(data: Dict[str, Any], target_key: str, train_or_val: str) \
        -> Tuple[List[float], List[int]]:
    """
        Extracts specific values from a dictionary based on a key. If train_or_val is 'train',
        then the training data will be returned, if train_or_val is 'val', then the validation data will be returned.

        Parameters:
        - data (Dict[str, Any]): Dictionary containing the data.
        - target_key (str): Key indicating which value to extract from `data`.
        - train_or_val (str): Indicates whether to extract data from 'train' or 'val' subset.

        Returns: The data and the associated epoch in a list.

        Raises:
        - ValueError: If `target_key` is not found in `data`, or if `train_or_val` is `None` when required.
    """
    epoch = data['epoch']
    if target_key == 'train_avg_loss':
        return [data['train_avg_loss']], [epoch]
    elif target_key == 'valid_avg_loss':
        return [data['valid_avg_loss']], [epoch]
    elif target_key == 'loss_classifier':
        assert train_or_val is not None, "If you want to access 'loss_classifier', train_or_val cannot be None"
        if train_or_val == 'train':
            return [data['epoch_all_other_avg_losses_per_epoch_train']['loss_classifier']], [epoch]
        else:
            return [data['epoch_all_other_avg_losses_per_epoch_val']['loss_classifier']], [epoch]
    elif target_key == 'loss_box_reg':
        assert train_or_val is not None, "If you want to access 'loss_box_reg', train_or_val cannot be None"
        if train_or_val == 'train':
            return [data['epoch_all_other_avg_losses_per_epoch_train']['loss_box_reg']], [epoch]
        else:
            return [data['epoch_all_other_avg_losses_per_epoch_val']['loss_box_reg']], [epoch]
    elif target_key == 'loss_mask':
        assert train_or_val is not None, "If you want to access 'loss_mask', train_or_val cannot be None"
        if train_or_val == 'train':
            return [data['epoch_all_other_avg_losses_per_epoch_train']['loss_mask']], [epoch]
        else:
            return [data['epoch_all_other_avg_losses_per_epoch_val']['loss_mask']], [epoch]
    elif target_key == 'loss_objectness':
        assert train_or_val is not None, "If you want to access 'loss_objectness', train_or_val cannot be None"
        if train_or_val == 'train':
            return [data['epoch_all_other_avg_losses_per_epoch_train']['loss_objectness']], [epoch]
        else:
            return [data['epoch_all_other_avg_losses_per_epoch_val']['loss_objectness']], [epoch]
    elif target_key == 'loss_rpn_box_reg':
        assert train_or_val is not None, "If you want to access 'loss_rpn_box_reg', train_or_val cannot be None"
        if train_or_val == 'train':
            return [data['epoch_all_other_avg_losses_per_epoch_train']['loss_rpn_box_reg']], [epoch]
        else:
            return [data['epoch_all_other_avg_losses_per_epoch_val']['loss_rpn_box_reg']], [epoch]
    elif target_key == 'learning_rate':
        return [data['learning_rate']], [epoch]
    elif target_key == 'model_architecture':
        return [data['model_architecture']], [epoch]
    else:
        raise ValueError("Key not in dictionary")


def parse_training_metadata_json(file_path: str, key: str) -> Union[List[int], str, int, float]:
    """
        Parse a JSON file and retrieve a value associated with a given key.

        Parameters:
        - file_path (str): Path to the JSON file to be parsed.
        - key (str): Key whose corresponding value is to be retrieved from the JSON data.

        Returns: Data for the key.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if key in data:
                return data[key]
            else:
                return f"Key '{key}' not found in JSON data."
    except FileNotFoundError:
        return f"File '{file_path}' not found."
    except json.JSONDecodeError as e:
        return f"Error parsing JSON in '{file_path}': {e}"


if __name__ == '__main__':
    print(sys.float_info.epsilon)
    # {"epoch": 6, "train_loss": 0.6169852428138256, "valid_loss": 0.524792030453682,
    # "epoch_all_other_avg_losses_per_epoch_train": {"loss_classifier": 0.045109601225703955, "loss_box_reg": 0.0718561951071024,
    # "loss_mask": 0.49009718373417854, "loss_objectness": 0.006426771113183349, "loss_rpn_box_reg": 0.0034954891307279468},
    # "epoch_all_other_avg_losses_per_epoch_val": {"loss_classifier": 0.05190129578113556, "loss_box_reg": 0.056991782039403915,
    # "loss_mask": 0.4072853624820709, "loss_objectness": 0.00698393490165472, "loss_rpn_box_reg": 0.001629688762477599},
    # "evaluation_dict_val": {"avg_box_iou": 0.5021514892578125, "avg_mask_iou": 0, "avg_box_precision": 0.08333333333333333, "avg_box_recall": 0.5,
    # "avg_mask_precision": 0.0, "avg_mask_recall": 0.0, "avg_f1_boxes": 0.14285711836735113, "avg_f1_masks": 0.0},
    # "learning_rate": 8.044231028658248e-07, "model_architecture": "maskrcnn_resnet50_fpn_v2"}
    # only can get one key at a time for clarity
    # print(
    #     parse_training_results_json("mask_rcnn_models/test/training_metadata_all_losses.json", 'valid_avg_loss', 'val'))
