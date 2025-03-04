import os
from itertools import zip_longest
from typing import List, Optional, Dict
import numpy as np
import torch
import albumentations as A
from albumentations import BaseCompose
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from utils import validate_targets, \
    unpack_stockpile_dataset_elements, select_n_random, create_bounding_box_tensor, create_mask_tensor, \
    apply_transform_without_annotations, apply_transformation_with_optional_workaround, \
    convert_float32_tensor_to_uint8_tensor, get_model_prediction_as_dict, plot_comparison_images, \
    load_mask_rcnn_on_device, parse_training_results_json, generate_n_distinct_colors


def display_image_tensors(
        elements: torch.Tensor,
        masks: Optional[List[torch.Tensor]] = None,
        bboxes: Optional[List[torch.Tensor]] = None,
        labels: Optional[List[torch.Tensor]] = None,
        label_map: Optional[Dict[int, str]] = None,
        titles: Optional[List[str]] = None,
        cols: int = 4,
        font_size_title: int = 50,
        masks_alpha: float = 0.3,
        bbox_width: int = 7,
        bbox_font_size: int = 60,
        font_file_path: str = "resources/Font/Roboto-Regular.ttf"
) -> None:
    """
    Display images along with optional masks and bounding boxes.

    Args:
        elements (torch.Tensor): Input images tensor of shape [N, C, H, W].
            N ist the number of images and must be a torch.uint8 or torch.float32 tensor.
        masks (Optional[List[torch.Tensor]): List of masks. Each element is a
            [N, H ,W] torch.uint8 tensor, where N is number of instances on the image.
        bboxes (Optional[List[torch.Tensor]]): List of bounding boxes. Each element is a  [N, 4] torch.float32 tensor,
            where N is number of instances on the image.
        labels (Optional[List[torch.Tensor]]): List of labels. Each element is a [N] torch.int64 tensor,
            where N is number of instances on the image.
        label_map (Optional[Dict[int, str]]): Dictionary mapping label indices to label names. Defaults to None.
        titles (Optional[List[str]]): Titles for each image. Defaults to None.
        cols (int): Number of columns in the grid. Defaults to 4.
        font_size_title (int): Font size for titles. Defaults to 50.
        masks_alpha (float): Alpha value for masks. Defaults to 0.3.
        bbox_width (int): Width of the bounding box lines. Defaults to 7.
        bbox_font_size (int): Font size for bounding box labels. Defaults to 60.
        font_file_path (str): Path to the font file for titles. Defaults to "resources/Font/Roboto-Regular.ttf".

    Raises:
            ValueError: If `elements` is not a torch.float32 or torch.uint8 [N, C, H, W] tensor.
    """
    titles = [f"Image {str(i)}" for i in range(len(elements))] if titles is None or len(titles) != len(
        elements) else titles

    if isinstance(elements, torch.Tensor):
        number_images = elements.shape[0]
    else:
        raise ValueError("Images must be torch.float32 or torch.uint8 [N, C, H, W] tensor")

    rows = number_images // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))

    check_plot_masks = validate_targets(elements, target=masks)
    check_plot_bounding_boxes = validate_targets(elements, target=bboxes)
    check_plot_labels = validate_targets(elements, target=labels)

    for i, (element, title) in enumerate(zip(elements, titles)):

        plt.subplot(rows, cols, i + 1)
        plt.title(title, fontsize=font_size_title)
        plt.axis('off')

        element = convert_float32_tensor_to_uint8_tensor(element)

        # Masks for mask r-cnn are in uint8, plotting requires bool
        if check_plot_masks and masks[i].dtype == torch.uint8:
            masks[i] = masks[i].bool()

        color_seed = 7  # only matters that seed is same for masks and boxes
        annotated_tensor = element
        if check_plot_masks:
            annotated_tensor = create_mask_tensor(element=annotated_tensor, mask=masks[i],
                                                  mask_alpha=masks_alpha, color_seed=color_seed)

        if check_plot_bounding_boxes:
            labels_bbox = labels[i] if check_plot_labels is not False else None
            annotated_tensor = create_bounding_box_tensor(element=annotated_tensor, bbox=bboxes[i],
                                                          labels=labels_bbox, label_map=label_map,
                                                          color_seed=color_seed,
                                                          bbox_width=bbox_width, bbox_font_size=bbox_font_size,
                                                          bbox_font_file_path=font_file_path)

        img_np = annotated_tensor.numpy()
        plt.imshow(img_np.transpose((1, 2, 0)))

    plt.tight_layout()
    plt.show()


def display_transforms_with_annotations_with_optional_workaround(
        elements: torch.Tensor,
        transforms: BaseCompose,
        masks: List[torch.Tensor],
        bboxes: List[torch.Tensor],
        labels: List[torch.Tensor],
        label_map: Dict[int, str],
        use_workaround: bool = False,
        mask_alpha: float = 0.3,
        bbox_width: int = 7,
        bbox_font_size: int = 60,
        font_file_path: str = "resources/Font/Roboto-Regular.ttf"
) -> None:
    """
    Display images before and after applying transformations with annotations, with an optional workaround
    for bugs in Albumentations. See `utils.apply_transform_with_annotations_with_workaround()` or
    `stockpile_dataset._apply_albumentations_transforms()` for more information.

    Args:
        elements (torch.Tensor): Input images of shape [N, C, H, W]. N ist the number of images.
            Must be a torch.float32 or torch.uint8 tensor.
        transforms (BaseCompose): Albumentations transformation function to apply to each image.
        masks (List[torch.Tensor]):  List of  masks for each image.
            Each element is a [N, H ,W] torch.uint8 tensor, where N is number of instances on the image.
        bboxes (List[torch.Tensor]): List of bounding boxes for each image.
            Each element is a [N, 4] torch.float32 tensor, where N is number of instances on the image.
        labels (List[torch.Tensor]): List of labels for each image.
            Each element is a [N] torch.int64 tensor, where N is number of instances on the image.
        label_map (Dict[int, str]): Dictionary mapping label indices to label names.
        use_workaround (bool): Whether to use a workaround for transformations that do not support annotations. Defaults to False.
        mask_alpha (float): Alpha value for masks. Defaults to 0.3.
        bbox_width (int): Width of the bounding box lines. Defaults to 7.
        bbox_font_size (int): Font size for the bounding box labels. Defaults to 60.
        font_file_path (str): Path to the font file for bounding box labels. Defaults to "resources/Font/Roboto-Regular.ttf".

    Raises:
            ValueError: If `elements` is not a torch.float32 or torch.uint8 [N, C, H, W] tensor.
    """

    if isinstance(elements, torch.Tensor):
        num_images = elements.shape[0]
    else:
        raise ValueError("Images must be torch.float32 or torch.uint8 [N, C, H, W] tensor")

    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
    for i, (element, mask, bbox, label) in enumerate(zip_longest(elements, masks, bboxes, labels, fillvalue=None)):
        color_seed = 7

        annotated_tensor_transformed, annotation_dict = apply_transformation_with_optional_workaround(
            element, transforms, mask=mask, bbox=bbox, label=label, use_workaround=use_workaround)

        annotated_tensor_original = convert_float32_tensor_to_uint8_tensor(element)
        annotated_tensor_transformed = convert_float32_tensor_to_uint8_tensor(annotated_tensor_transformed)

        annotated_tensor_original = create_mask_tensor(annotated_tensor_original, mask=mask.bool(),
                                                       mask_alpha=mask_alpha, color_seed=color_seed)
        annotated_tensor_transformed = create_mask_tensor(annotated_tensor_transformed,
                                                          mask=annotation_dict['masks'].bool(),
                                                          mask_alpha=mask_alpha, color_seed=color_seed)

        annotated_tensor_original = create_bounding_box_tensor(element=annotated_tensor_original, bbox=bbox,
                                                               labels=label, label_map=label_map,
                                                               color_seed=color_seed,
                                                               bbox_width=bbox_width, bbox_font_size=bbox_font_size,
                                                               bbox_font_file_path=font_file_path)

        annotated_tensor_transformed = create_bounding_box_tensor(element=annotated_tensor_transformed,
                                                                  bbox=annotation_dict['boxes'],
                                                                  labels=annotation_dict['labels'],
                                                                  label_map=label_map,
                                                                  color_seed=color_seed,
                                                                  bbox_width=bbox_width,
                                                                  bbox_font_size=bbox_font_size,
                                                                  bbox_font_file_path=font_file_path)

        original_np = annotated_tensor_original.numpy().transpose((1, 2, 0))
        transformed_np = annotated_tensor_transformed.numpy().transpose((1, 2, 0))
        plot_comparison_images(axes, num_images, i, original_np, transformed_np,
                               f"Image {i} Original", f"Image {i} Model Prediction")

    plt.tight_layout()
    plt.show()


def display_transforms_without_annotation(elements: torch.Tensor, transforms: BaseCompose):
    """
        Display images before and after applying transformations without annotations.

        Args:
            elements (torch.Tensor): Input images of shape [N, C, H, W]. Must be a torch.float32 or torch.uint8 tensor.
                N ist the number of images.
            transforms (BaseCompose): Albumentations transformation function to apply to each image.

        Raises:
            ValueError: If `elements` is not a torch.float32 or torch.uint8 [N, C, H, W] tensor.
    """
    if isinstance(elements, torch.Tensor):
        num_images = elements.shape[0]
    else:
        raise ValueError("Images must be torch.float32 or torch.uint8 [N, C, H, W] tensor")

    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
    for i, element in enumerate(elements):
        element = convert_float32_tensor_to_uint8_tensor(element)

        annotated_tensor_original = element
        annotated_tensor_transformed = apply_transform_without_annotations(element.clone(), transforms)

        original_np = annotated_tensor_original.numpy().transpose((1, 2, 0))
        transformed_np = annotated_tensor_transformed.numpy().transpose((1, 2, 0))

        plot_comparison_images(axes, num_images, i, original_np, transformed_np,
                               f"Image {i} Original", f"Image {i} Model Prediction")

    plt.tight_layout()
    plt.show()


# images can only be [N, C, H, W] tensor
def display_maskrcnn_predictions_and_ground_truth(
        images_gt: torch.Tensor,
        masks_gt: List[torch.Tensor],
        bboxes_gt: List[torch.Tensor],
        labels_gt: List[torch.Tensor],
        label_map: Dict[int, str],
        path_model: str,
        trainable_layers: int,
        number_of_classes: int,
        mode: str = 'rgb',
        dataset=None,
        threshold: float = 0.5,
        bbox_width: int = 7,
        bbox_font_size: int = 60,
        masks_alpha: float = 0.3,
        font_file_path: str = "resources/Font/Roboto-Regular.ttf",
        device: torch.device = torch.device("cpu")
) -> None:
    """
    Display Mask R-CNN predictions and ground truth images side by side.

    Args:
        images_gt (torch.Tensor): Ground truth images of shape [N, C, H, W].
            N ist the number of images and must be a torch.uint8 or torch.float32 tensor.
        masks_gt (List[torch.Tensor]): List of ground truth masks for each image. Each element is a
            [N, H ,W] torch.uint8 tensor, where N is number of instances on the image.
        bboxes_gt (List[torch.Tensor]): List of ground truth bounding boxes for each image.
            Each element is a [N, 4] torch.float32 tensor, where N is number of instances on the image.
        labels_gt (List[torch.Tensor]): List of ground truth labels for each image.
            Each element is a [N] torch.int64 tensor, where N is number of instances on the image.
        label_map (Dict[int, str]): Dictionary mapping label indices to label names.
        path_model (str): Path to the Mask R-CNN model.
        trainable_layers (int): Number of trainable layers in the loaded Mask R-CNN model.
        number_of_classes (int): Number of classes in the dataset used by the Mask R-CNN model.
        mode (str, optional): Color mode for images ('rgb', 'rgbh', or 'rgb_blend'). Default is 'rgb'.
        dataset (optional): Dataset object used to obtain mean and standard deviation when 'rgbh' is used.
        threshold (float): Threshold for model predictions.
        bbox_width (int): Width of the bounding box lines.
        bbox_font_size (int): Font size for the bounding box labels.
        masks_alpha (float): Alpha value for masks.
        font_file_path (str): Path to the font file for bounding box labels.
        device (torch.device): GPU ('cuda') or CPU ('cpu').
     Raises:
            ValueError: If `elements` is not a torch.float32 or torch.uint8 [N, C, H, W] tensor.
    """
    if isinstance(images_gt, torch.Tensor):
        num_images = images_gt.shape[0]
    else:
        raise ValueError("Images must be torch.float32 or torch.uint8 [N, C, H, W] tensor")

    model = load_mask_rcnn_on_device(path_model, device=device, trainable_layers=trainable_layers, mode=mode,
                                     dataset=dataset, number_of_classes=number_of_classes)
    model.eval()

    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
    plt.axis()

    # get model predictions for everything
    model_prediction = get_model_prediction_as_dict(model, images_gt, threshold)
    model_masks = model_prediction['masks']
    model_bboxes = model_prediction['bboxes']
    model_labels = model_prediction['labels']

    for i, (image_gt, mask_gt, bbox_gt, label_gt, model_mask, model_bbox, model_label) in (
            enumerate(
                zip(images_gt[:, :3, :, :], masks_gt, bboxes_gt, labels_gt, model_masks, model_bboxes, model_labels))):
        annotated_tensor_original = convert_float32_tensor_to_uint8_tensor(image_gt.detach().clone())
        annotated_tensor_model = convert_float32_tensor_to_uint8_tensor(image_gt.detach().clone())
        color_seed = 7

        annotated_tensor_original = create_mask_tensor(element=annotated_tensor_original, mask=mask_gt.bool(),
                                                       mask_alpha=masks_alpha, color_seed=color_seed)

        annotated_tensor_model = create_mask_tensor(element=annotated_tensor_model, mask=model_mask,
                                                    mask_alpha=masks_alpha, color_seed=color_seed)

        annotated_tensor_original = create_bounding_box_tensor(element=annotated_tensor_original, bbox=bbox_gt,
                                                               labels=label_gt, label_map=label_map,
                                                               color_seed=color_seed,
                                                               bbox_width=bbox_width, bbox_font_size=bbox_font_size,
                                                               bbox_font_file_path=font_file_path)

        annotated_tensor_model = create_bounding_box_tensor(element=annotated_tensor_model,
                                                            bbox=model_bbox,
                                                            labels=model_label,
                                                            label_map=label_map,
                                                            color_seed=color_seed,
                                                            bbox_width=bbox_width,
                                                            bbox_font_size=bbox_font_size,
                                                            bbox_font_file_path=font_file_path)

        original_np = annotated_tensor_original.numpy().transpose((1, 2, 0))
        model_np = annotated_tensor_model.numpy().transpose((1, 2, 0))
        plot_comparison_images(axes, num_images, i, original_np, model_np,
                               f"Image {i} Original", f"Image {i} Model Prediction")

    plt.tight_layout()
    plt.show()


def display_rgbh_tensors(elements: torch.Tensor,
                         masks: Optional[List[torch.Tensor]] = None,
                         bboxes: Optional[List[torch.Tensor]] = None,
                         labels: Optional[List[torch.Tensor]] = None,
                         label_map: Optional[Dict[int, str]] = None,
                         n_cols: int = 2,
                         masks_alpha: float = 0.3,
                         bbox_width: int = 7,
                         bbox_font_size: int = 60,
                         font_file_path: str = "resources/Font/Roboto-Regular.ttf"
                         ):
    """
        Display RGB and H tensors side by side, optionally overlaying masks and bounding boxes on the RGB Tensor.

        Args:
            elements (torch.Tensor): Tensor containing both RGB and H channels of shape [N, 4, H, W].
                N is the number of images. The first 3 channels are RGB, and the fourth channel is H.
                Must be a torch.float32 tensor.
            masks (Optional[List[torch.Tensor]]): List of masks for each image.
                Each element is a [N, H, W] torch.Tensor, where N is the number of instances on the image.
                Masks should be torch.uint8 tensors. Default is None (no masks are plotted).
            bboxes (Optional[List[torch.Tensor]]): List of bounding boxes for each image.
                Each element is a [N, 4] torch.Tensor, where N is the number of instances on the image.
                The 4 values represent [x_min, y_min, x_max, y_max]. Default is None (no bounding boxes are plotted).
            labels (Optional[List[torch.Tensor]]): List of labels for each image.
                Each element is a [N] torch.Tensor containing integer indices corresponding to label names in `label_map`.
                Default is None (no labels are plotted).
            label_map (Optional[Dict[int, str]]): Dictionary mapping label indices to label names.
                Default is None.
            n_cols (int): Number of columns in the plot grid. Default is 2.
            masks_alpha (float): Transparency (alpha value) of masks overlaid on images. Default is 0.3.
            bbox_width (int): Width of the bounding box lines in pixels. Default is 7.
            bbox_font_size (int): Font size for the bounding box labels. Default is 60.
            font_file_path (str): Path to the font file used for bounding box labels. Default is "resources/Font/Roboto-Regular.ttf".

        Raises:
            ValueError: If `elements` is not a torch.float32 [N, 4, H, W] tensor.
        """
    num_images = elements.shape[0]
    rgb_tensors = elements[:, :3, :, :]
    h_tensors = elements[:, 3:4, :, :]

    fig, axs = plt.subplots(num_images, n_cols, figsize=(10, num_images * 5))
    plt.axis()
    check_plot_masks = validate_targets(elements, target=masks)
    check_plot_bounding_boxes = validate_targets(elements, target=bboxes)
    check_plot_labels = validate_targets(elements, target=labels)

    for i, (rgb_tensor, h_tensor) in enumerate(zip(rgb_tensors, h_tensors)):
        rgb_tensor = convert_float32_tensor_to_uint8_tensor(rgb_tensor)

        # Masks for mask r-cnn are in uint8, plotting requires bool
        if check_plot_masks and masks[i].dtype == torch.uint8:
            masks[i] = masks[i].bool()

        color_seed = 7  # only matters that seed is same for masks and boxes
        annotated_tensor = rgb_tensor
        if check_plot_masks:
            annotated_tensor = create_mask_tensor(element=annotated_tensor, mask=masks[i],
                                                  mask_alpha=masks_alpha, color_seed=color_seed)

        if check_plot_bounding_boxes:
            labels_bbox = labels[i] if check_plot_labels is not False else None
            annotated_tensor = create_bounding_box_tensor(element=annotated_tensor, bbox=bboxes[i],
                                                          labels=labels_bbox, label_map=label_map,
                                                          color_seed=color_seed,
                                                          bbox_width=bbox_width, bbox_font_size=bbox_font_size,
                                                          bbox_font_file_path=font_file_path)

        img_np = annotated_tensor.numpy().transpose((1, 2, 0))
        h_tensor_np = h_tensor.numpy().transpose((1, 2, 0))
        plot_comparison_images(axs, num_images, i, img_np, h_tensor_np, f"RGB Tensor {i}",
                               f"H Tensor {i}")
    plt.tight_layout()  # Adjust layout to make it look better
    plt.show()


def visualise_training_results(json_file_path: str, target_keys: List[str], train_or_val: Optional[str], title: str,
                               y_label: str):
    """
     Visualize training results from a JSON file for specified target keys over epochs.

     Args:
         json_file_path (str): Path to the JSON file containing training results.
         target_keys (List[str]): List of keys to extract from the JSON data and visualize.
         train_or_val (Optional[str]): Specify 'train' or 'val' to select training or validation results.
         title (str): Title of the plot.
         y_label (str): Label for the y-axis.

     Raises:
         FileNotFoundError: If the specified `json_file_path` does not exist.
         ValueError: If `train_or_val` is provided but not 'train' or 'val'.
    """
    values_list = []
    epochs_list = []
    for target in target_keys:
        parse_json = parse_training_results_json(json_file_path, target, train_or_val)
        values_list.append(parse_json[0])
        epochs_list.append(parse_json[1])

    plt.figure(figsize=(10, 6))

    for i, (epochs, values) in enumerate(zip(epochs_list, values_list)):
        plt.plot(epochs, values, marker='o', linestyle='-', label=target_keys[i])

    # Add title and labels
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)

    # Add grid
    plt.grid(True)
    # Add legend
    plt.legend()
    # Show the plot
    plt.show()


def visualise_merging_results(path_np_array: str):
    """
        Visualize a merged numpy array with distinct colors representing unique instances of
        sand heaps.

        Args:
            path_np_array (str): Path to the numpy array.

        Raises:
            ValueError: If the input `path_np_array` is not a valid path or numpy array.
    """
    merged_np_array = np.load(path_np_array)
    unique_values = np.unique(merged_np_array)
    colors = [*generate_n_distinct_colors(len(unique_values) - 1)]
    colors_normalized = [(r / 255, g / 255, b / 255) for r, g, b in colors]
    colors_normalized.insert(0, (1, 1, 1))

    # Create a ListedColormap directly from the colors
    cmap = mcolors.ListedColormap(colors_normalized)

    # Set normalization boundaries to cover each unique value
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(unique_values) + 1), ncolors=len(unique_values))

    plt.figure(figsize=(10, 10))
    plt.imshow(merged_np_array, cmap=cmap, norm=norm, interpolation='none')
    plt.colorbar(ticks=np.arange(len(unique_values)))
    plt.show()


if __name__ == '__main__':
    from stockpile_dataset import StockpileDataset

    # visualise_training_results("mask_rcnn_models/Phase_1/trained_model_rgb_blend_512_512/training_metadata_all_losses.json",
    #                            ['valid_avg_loss', 'train_avg_loss'],
    #                            train_or_val='train', title='Average Loss per Epoch', y_label='Loss')
    #
    train_dataset = (StockpileDataset.create_builder(dataset_mode='test', image_mode='rgb_blend')
                     .set_has_been_built(True)).build()

    # train_augs = A.Compose([
    #     A.D4(p=1.0)
    # ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_classes"]))

    # train_dataset.set_transforms(train_augs)

    # #
    # # # train_dataset.set_transforms(train_augs)
    # #
    # # # Pick elements from dataset
    #
    elements, masks, bboxes, labels, image_indices = unpack_stockpile_dataset_elements(
        select_n_random(train_dataset, 15), mode='rgb_blend')

    # elements, masks, bboxes, labels, image_indices = unpack_stockpile_dataset_elements([train_dataset[78]],
    #                                                                                    mode='rgb')

    # display_rgbh_tensors(elements, masks, bboxes, labels,
    #                      label_map=train_dataset.label_map,
    #                      font_file_path="resources/Font/Roboto-Regular.ttf")

    #
    # display_maskrcnn_predictions_and_ground_truth(elements, masks, bboxes, labels, train_dataset.label_map,
    #                                               trainable_layers=2, number_of_classes=2,
    #                                               path_model="inference/inputs/inference_model/mask_rcnn_resnet50_fpn.pt")

    # #
    # display_transforms_with_annotations_with_optional_workaround(elements=elements, transforms=train_augs,
    #                                                              masks=masks, bboxes=bboxes,
    #                                                              labels=labels, label_map=train_dataset.label_map,
    #                                                              use_workaround=False)
    #
    # display_transforms_without_annotation(elements=elements, transforms=train_augs, transforms_seed=None)
    #
    # # Display elements
    # display_image_tensors(elements, masks=masks, bboxes=bboxes, labels=labels, masks_alpha=0.4,
    #                       label_map=train_dataset.label_map, cols=1, font_file_path="resources/Font/Roboto-Regular.ttf")
    #

    # tensor_files = [os.path.join("inference/orthophoto_tensors", f) for f in
    #                 os.listdir("inference/orthophoto_tensors") if
    #                 f.endswith('.pt')]
    #
    # tensor_files = sorted(tensor_files, key=lambda x: int(os.path.basename(os.path.splitext(x)[0])))
    #
    # tensors = []
    #
    # for tensor_file in tensor_files[:20]:
    #     tensor_path = os.path.join(tensor_file)
    #     tensor = torch.load(tensor_path)
    #     tensors.append(tensor)
    #
    # tensors = torch.stack(tensors)

    display_image_tensors(elements, masks=masks, label_map=train_dataset.label_map, cols=1,
                              font_file_path="resources/Font/Roboto-Regular.ttf")
