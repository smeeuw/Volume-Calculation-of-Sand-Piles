import datetime
from typing import Optional

import torch
from albumentations import BaseCompose
from torch import nn
from torch.nn import BatchNorm2d, ReLU, Dropout, init
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, maskrcnn_resnet50_fpn
from stockpile_dataset import StockpileDataset


# https://christianjmills.com/posts/pytorch-train-mask-rcnn-tutorial/
# with some adjustments, mainly for loading and storing
def initialise_maskrcnn(device: torch.device, trainable_backbone_layers: int, number_of_classes: int,
                        dtype_device=torch.float32) -> maskrcnn_resnet50_fpn:
    """
        Initialize and configure a Mask R-CNN model based on ResNet-50 FPN architecture.

        Args:
            device (torch.device): Device where the model will be loaded (CPU or CUDA).
            trainable_backbone_layers (int): Number of trainable layers in the backbone network.
            number_of_classes (int): Number of object classes (including background).
            dtype_device (torch.dtype, optional): Data type to use for model parameters (default: torch.float32).

        Returns:
            maskrcnn_resnet50_fpn: Initialized Mask R-CNN model based on ResNet-50 FPN architecture.

    """
    model: maskrcnn_resnet50_fpn_v2
    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT', trainable_backbone_layers=trainable_backbone_layers)

    # Get the number of input features for the classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # Get the number of output channels for the Mask Predictor
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

    # Replace the box predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box,
                                                      num_classes=number_of_classes)

    # Replace the mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced,
                                                       num_classes=number_of_classes)

    # Set the model's device and data type
    model.to(device, dtype=dtype_device)

    # Add attributes to store the device and model name for later reference
    model.device = device
    model.name = 'maskrcnn_resnet50_fpn_v2'

    return model


def initialise_maskrcnn_rgbh(device: torch.device, dataset: StockpileDataset,
                             trainable_backbone_layers: int, number_of_classes: int,
                             dtype_device: torch.dtype = torch.float32) -> maskrcnn_resnet50_fpn_v2:
    """
        Initialize Mask R-CNN model with ResNet-50 FPN backbone and adapt it for RGBH images.

        Args:
            device (torch.device): Device where the model will be loaded (CPU or CUDA).
            dataset: StockpileDataset containing method calculate_mean_std_height() to compute mean and std.
            trainable_backbone_layers (int): Number of trainable layers in the backbone network.
            number_of_classes (int): Number of object classes (including background).
            dtype_device (torch.dtype, optional): Data type to use for model parameters (default: torch.float32).
        Returns:
            maskrcnn_resnet50_fpn_v2: Initialized Mask R-CNN model with ResNet-50 FPN architecture adapted for RGBH images.

    """

    mean, std = dataset.calculate_mean_std_height()

    # Set normalization for fourth channel, rgb are from backbone
    # of ImageNet https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py
    mean_list = [0.485, 0.456, 0.406, mean]
    std_list = [0.229, 0.224, 0.225, std]
    model: maskrcnn_resnet50_fpn_v2
    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT', trainable_backbone_layers=trainable_backbone_layers,
                                     image_mean=mean_list, image_std=std_list)

    # Get the number of input features for the classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # Get the number of output channels for the Mask Predictor
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

    # Replace the box predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box,
                                                      num_classes=number_of_classes)

    # Replace the mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced,
                                                       num_classes=number_of_classes)

    # Adjust first convolutional layer to accept 4 channel input
    body_resnet = model.backbone.body

    weights_rgb_conv1 = body_resnet.conv1.weight.clone()

    body_resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        body_resnet.conv1.weight[:, :3] = weights_rgb_conv1
        init.kaiming_normal_(body_resnet.conv1.weight[:, 3:4], nonlinearity='relu')

    # Set the model's device and data type
    model.to(device, dtype=dtype_device)

    # Add attributes to store the device and model name for later reference
    model.device = device
    model.name = 'maskrcnn_resnet50_fpn_v2'

    return model


def change_layers_to_eval(model: nn.Module):
    """
       Switches BatchNorm, ReLu, Dropout Layers to Evaluation mode.

       Args:
           model (nn.Module): The PyTorch model whose specific modules will be switched to evaluation mode.
    """
    model.train()  # we keep training
    # but we change the modules that behave differently to eval
    for name, module in model.named_modules():
        if isinstance(module, (_BatchNorm, ReLU, Dropout)):
            module.eval()


def setup_datasets_training(image_mode: str,
                            augmentations_train: Optional[BaseCompose] = None,
                            augmentations_val: Optional[BaseCompose] = None,
                            datasets_have_been_built: bool = False) -> tuple[StockpileDataset, StockpileDataset]:
    """
       Set up training and validation datasets for a specific image mode with optional augmentations.

       Args:
           image_mode (str): Mode of the images, must be one of ['rgb', 'rgb_blend', 'rgbh'].
           augmentations_train (BaseCompose, optional): Augmentations to apply to the training dataset (default: None).
           augmentations_val (BaseCompose, optional): Augmentations to apply to the validation dataset (default: None).
           datasets_have_been_built (bool, optional): Flag indicating if datasets have already been built
            (default: False).

       Returns:
           tuple[StockpileDataset, StockpileDataset]: A tuple containing the training dataset and validation dataset.

       Raises:
           AssertionError: If `image_mode` is not one of ['rgb', 'rgb_blend', 'rgbh'].

    """
    assert image_mode in ['rgb', 'rgb_blend', 'rgbh'], "image_mode must be 'rgb' or 'rgb_blend' or 'rgbh'."

    # Build the datasets
    train_dataset = (StockpileDataset.create_builder(dataset_mode='train', image_mode=image_mode)
                     .set_has_been_built(datasets_have_been_built)).build()

    val_dataset = (StockpileDataset.create_builder(dataset_mode='val', image_mode=image_mode)
                   .set_has_been_built(datasets_have_been_built)).build()

    # Set the augmentations for the datasets
    if augmentations_train is not None:
        train_dataset.set_transforms(augmentations_train)

    if augmentations_val is not None:
        val_dataset.set_transforms(augmentations_val)

    return train_dataset, val_dataset


def setup_training_metadata(image_mode: str,
                            image_width: int,
                            image_height: int,
                            trained_backbone_layers: int,
                            batch_size: int,
                            lr: float,
                            epochs: int,
                            early_stop_number_epochs: int,
                            number_classes: int,
                            blend_alpha: Optional[float] = None) -> dict:
    """
    Set up metadata dictionary for training configuration.

    Args:
        image_mode (str): Mode of the images, must be one of ['rgb', 'rgb_blend', 'rgbh'].
        image_width (int): Width of the input images.
        image_height (int): Height of the input images.
        trained_backbone_layers (int): Number of trainable layers in the backbone network.
        batch_size (int): Batch size used for training.
        lr (float): The learning rate.
        epochs (int): Number of epochs for training.
        early_stop_number_epochs (int): Number of epochs to wait before early stopping if no validation improvement.
        number_classes (int): Number of object classes (including background).
        blend_alpha (Optional[float]): Alpha value for blending in case of 'rgb_blend' mode (default: None).

    Returns:
        dict: Dictionary containing training configuration metadata.

    Raises:
        AssertionError: If `image_mode` is not one of ['rgb', 'rgb_blend', 'rgbh'].

    """
    assert image_mode in ['rgb', 'rgb_blend', 'rgbh'], "image_mode must be 'rgb' or 'rgb_blend' or 'rgbh'."

    train_parameters = {
        'mode': image_mode,
        'image_dimensions': [image_width, image_height],
        'trained_backbone_layers': trained_backbone_layers,
        'number_classes': number_classes,
        'batch_size': batch_size,
        'max_lr': lr,
        'initial_epochs': epochs,
        'early_stop_after': early_stop_number_epochs,
        'start_time': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }

    if image_mode == 'rgb_blend':
        train_parameters.update({'blend_alpha': blend_alpha})

    return train_parameters


if __name__ == '__main__':
    print(torch.optim.AdamW.__name__)
