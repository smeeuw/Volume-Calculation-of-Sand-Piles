import os
from typing import Dict, Tuple, Union, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from utils import convert_bbox_list_to_n4_tensor, convert_label_list_to_n_tensor, \
    convert_uint8_tensor_to_float32_tensor
from albumentations.core.composition import BaseCompose


class StockpileDatasetIterator:
    """
    Iterator for iterating over a StockpileDataset.

    This class provides an iterator interface for iterating over a StockpileDataset instance. It allows iterating
    over the dataset to access each item sequentially.

    Attributes:
        dataset: The StockpileDataset instance to iterate over.
        current (int): The index of the current item being iterated.
    """

    def __init__(self, dataset):
        """
        Initialize the StockpileDatasetIterator.

        Args:
            dataset: The StockpileDataset instance to iterate over.
        """
        self.dataset = dataset
        self.current = 0

    def __iter__(self):
        """
        Return the iterator object.

        Returns:
            StockpileDatasetIterator: The iterator object.
        """
        return self

    def __next__(self):
        """
        Return the next item from the dataset.

        Returns:
            Any: The next item from the dataset.

        Raises:
            StopIteration: If there are no more items to iterate over.
        """
        if self.current < len(self.dataset):
            item = self.dataset[self.current]
            self.current += 1
            return item
        else:
            raise StopIteration


class StockpileDataset(Dataset):
    """
    A dataset class for stockpile analysis. An instance of this class should only be generated with the builder.
    The builder can be received by calling `create_builder()`.

    Args:
        image_mode (str): The image mode of the dataset. Can be 'rgb', 'rgb_blend', or 'rgbh'.
        file_dictionary (dict[str,str]): A dictionary containing file paths for dataset files.
            The filenames are handled internally. This is for matching masks, bounding patches, orthophoto patches,
            and optionally DSM patches.
        length (int): The length of the dataset.
        label_map (dict[int, str]): A dictionary mapping integers to class labels.
        remove_mask_threshold (int): The minimum number of pixels to keep the mask and the corresponding
            bounding box and label annotations. If the mask takes fewer pixels than this threshold,
            the mask and corresponding bounding box and label annotations are removed (considered as noise).
        image_width(int): The width of the image (in pixels).
        image_height(int): The height of the image (in pixels).
    """

    @staticmethod
    def create_builder(dataset_mode: str, image_mode: str):
        """
        Create a StockpileDatasetBuilder instance. IMPORTANT: Only create an instance of this dataset with a builder!

        This static method creates and returns an instance of StockpileDatasetBuilder based on the specified mode and
        whether only RGB data is to be used.

        Args:
            dataset_mode (str): The mode of the dataset ("train", "val", "test").
            image_mode (bool): The image mode of the dataset. Can be 'rgb', 'rgb_blend', or 'rgbh'.

        Returns:
            StockpileDatasetBuilder: An instance of StockpileDatasetBuilder configured based on the specified mode and
            only_use_rgb parameter.
        """
        from stockpile_dataset_builder import StockpileDatasetBuilder

        assert image_mode in ['rgb', 'rgb_blend', 'rgbh']
        assert dataset_mode in ['train', 'val', 'test']

        return StockpileDatasetBuilder(dataset_mode, image_mode)

    def __init__(self, image_mode: str, file_dictionary: Dict[str, str], length: int, label_map: Dict[int, str],
                 remove_mask_threshold: int, image_width: int, image_height: int) -> None:
        """
        Initializes the StockpileDataset instance. IMPORTANT: Only create an instance of this dataset with a builder!

        Parameters:
            image_mode (str): The image mode of the dataset. Can be 'rgb', 'rgb_blend', or 'rgbh'.
            file_dictionary (Dict[str, str]): A dictionary containing file paths for dataset files.
            The filenames are handled internally. This is for matching masks, bounding patches, orthophoto patches,
            and optionally DSM patches.
            length (int): The length of the dataset.
            label_map (Dict[int, str]): A dictionary mapping integers to class labels.
            remove_mask_threshold (int): The minimum number of pixels to keep the mask and the corresponding
            bounding box and label annotations. If the mask takes fewer pixels than this threshold,
            the mask and corresponding bounding box and label annotations are removed (considered as noise).
            image_width (int): The image width in pixels.
            image_height (int): The image height in pixels.
        """
        super(Dataset, self).__init__()
        self.image_mode = image_mode
        self.file_dictionary = file_dictionary
        self.length = length
        self.remove_mask_threshold = remove_mask_threshold
        self.label_map = label_map
        self.image_width = image_width
        self.image_height = image_height
        assert len(self.label_map) == 2, ("Code must be adjusted. "
                                          "It is no longer a binary classification problem")

        self.transforms = None

    def __len__(self) -> int:
        """
            Returns the length of the dataset.
            Returns:
                int: The length of the dataset.
            """
        return self.length

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, int]]], Tuple[
        torch.Tensor, torch.Tensor, Dict[str, Union[torch.Tensor, int]]]]:
        """
            Retrieves the item at the specified index from the dataset and applies data augmentation, if specified
            with `set_transforms`. Returns a tuple containing the image tensor,
            the DSM tensor if `image_mode` is `rgbh`, and a dictionary with mask, bounding box, and label tensors.
            Hint: Check set_transform() for information about the requirements for successful augmentation.

            Args:
                index (int): The index of the item to retrieve.

            Returns:
                tuple: A tuple containing the image tensor, the DSM tensor if image_mode is `rgbh`, and a dictionary
                       with mask, bounding box, and label tensors.
            Notes:
                - Depending on the augmentation (`transforms`) the datatype of the returned image is either torch.uint8 or torch.float32.
                - The dictionary contains the following keys: 'masks', 'bounding_boxes', 'labels', 'image_id'.
                - The entry for the key 'masks' is a torch.unit8 [N, H, W] tensor.
                - The entry for the key 'bounding_boxes' is a torch.float32 [N,4] tensor in 'xyxy' format.
                - The entry for the key 'labels' is a torch.int64 scalar [N] tensor.
                - N is the number of masks on the image and also specifies the number of corresponding bounding boxes and labels.
                - The entry for the key 'image_id' gives back an integer, which is the filename of the image.
        """
        if self.image_mode == 'rgb_blend' or self.image_mode == 'rgb':
            ortho_tensor, masks_tensor, bboxes_tensor = self._load_tensors_from_file(index)
        else:
            ortho_tensor, dsm_tensor, masks_tensor, bboxes_tensor = self._load_tensors_from_file(index)

        # remove annotation of masks below threshold
        masks_tensor, bboxes_tensor = (self._remove_tensors_below_threshold
                                       (masks_tensor, bboxes_tensor))

        # binary classification problem, 1 for Stockpile
        labels_tensor = torch.ones(masks_tensor.shape[0], dtype=torch.int64)

        self._validate_input_mask_rcnn(ortho_tensor, masks_tensor, bboxes_tensor, labels_tensor)

        if self.image_mode == 'rgb_blend' or self.image_mode == 'rgb':
            image, target = ortho_tensor, {'masks': masks_tensor, 'boxes': bboxes_tensor, 'labels': labels_tensor,
                                           'image_id': index}
            if self.transforms is not None:
                return self._apply_albumentations_transforms_rgb(image, target)
            return image, target
        else:
            if self.transforms is not None:
                transformed = self.transforms(
                    image=ortho_tensor.numpy(force=True).transpose((1, 2, 0)),
                    image0=dsm_tensor.numpy(force=True).transpose((1, 2, 0)),
                    mask=masks_tensor.numpy(force=True).transpose((1, 2, 0)),
                    bboxes=bboxes_tensor.numpy(force=True),
                    bbox_classes=labels_tensor.numpy(force=True)
                )

                ortho_tensor_trans = convert_uint8_tensor_to_float32_tensor(
                    torch.tensor(transformed["image"].transpose((2, 0, 1))))
                dsm_tensor_trans = torch.tensor(transformed["image0"].transpose((2, 0, 1)))
                mask_tensor_trans = torch.tensor(transformed["mask"].transpose((2, 0, 1)))
                bboxes_tensor = convert_bbox_list_to_n4_tensor(transformed["bboxes"], dtype=torch.float32)
                labels_tensor = convert_label_list_to_n_tensor(transformed["bbox_classes"], dtype=torch.int64)
                self._validate_input_mask_rcnn(ortho_tensor_trans, mask_tensor_trans, bboxes_tensor, labels_tensor)

                return (torch.cat((ortho_tensor_trans, dsm_tensor_trans), dim=0),
                        {'masks': mask_tensor_trans, 'boxes': bboxes_tensor, 'labels': labels_tensor,
                        'image_id': index})

            ortho_tensor = convert_uint8_tensor_to_float32_tensor(ortho_tensor)
            image, target = torch.cat((ortho_tensor, dsm_tensor), dim=0), {'masks': masks_tensor,
                                                                           'boxes': bboxes_tensor,
                                                                           'labels': labels_tensor,
                                                                           'image_id': index}
            return image, target

    def __iter__(self) -> 'StockpileDatasetIterator':
        """
        Return an iterator for the StockpileDataset.

        This method returns an iterator object of type 'StockpileDatasetIterator' for the StockpileDataset instance.

        Returns:
            StockpileDatasetIterator: An iterator object for iterating over the StockpileDataset.
            """
        return StockpileDatasetIterator(self)

    def get_image_width(self):
        """
           Get the width of the images in the dataset.

           Returns:
               int: Width of the images in the dataset.
        """
        return self.image_width

    def get_image_height(self):
        """
        Get the height of the images in the dataset.

        Returns:
            int: Height of the images in the dataset.
        """
        return self.image_height

    def set_transforms(self, transforms: BaseCompose) -> None:
        """
        Set the Albumentations transforms for the dataset. IMPORTANT: When the mode is 'rgb' or 'rgb_blend'
        it is assumed, that the Albumentations augmentation also uses ToFloat(max_value=255.0)
        and ToTensorV2(transpose_mask=True), as the image of the dataset is always torch.uint8 without modification.
        When the mode is 'rgbh', it is expected that ToFloat() and ToTensorV2() is NOT used, since the
        normalisation would also affect the DSM, which is why it is handled internally.

        This method sets the Albumentations transforms to be applied to the dataset samples during data loading.

        Args:
            transforms (BaseCompose): The Albumentations transforms to be applied to the dataset samples.
        """
        if not isinstance(transforms, BaseCompose):
            raise Exception('Transforms is not an instance of BaseCompose. '
                            'Only Albumentations Augmentations are supported.')
        self.transforms = transforms

    def _remove_tensors_below_threshold(self, masks_tensor: torch.Tensor, bboxes_tensor: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method removes tensors from `masks_tensor` and `bboxes_tensor` when the number of pixels
        in a mask are below the threshold specified by `remove_mask_threshold`. It removes the annotations
        of the corresponding bounding boxes as well, effectively transforming this area to noise. Since the
        labels are generated afterwards in __getitem__, there will not be a mismatch.

        Args:
            masks_tensor (torch.Tensor): The tensor containing masks. This is a torch.uint8 [N, H, W] tensor.
                N is the number of masks (polygons) and corresponding bounding boxes on the image patch.
            bboxes_tensor (torch.Tensor): The tensor containing bounding boxes. This is a torch.float32 [N,4] tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing filtered `masks_tensor` and `bboxes_tensor`.
        """
        assert masks_tensor.dtype == torch.uint8, "Masks tensor should be in uint8 format."
        assert bboxes_tensor.dtype == torch.float32, "Bounding boxes should be in float32 format."
        assert self.image_height == masks_tensor.shape[1] and self.image_width == masks_tensor.shape[2], \
            "Mask must be of shape [N, H, W], where N is the number of instances on the image"
        mask_sums = masks_tensor.sum(dim=(1, 2))
        keep_indices = mask_sums >= self.remove_mask_threshold
        return masks_tensor[keep_indices], bboxes_tensor[keep_indices]

    def calculate_mean_std_rgb(self) -> Tuple[List[float], List[float]]:
        """
        Calculate mean and standard deviation of an RGB dataset. If an RGBH dataset is provided, only the first
        3 channels will be considered.

        This method calculates the mean and standard deviation of the dataset by iterating over its samples.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing lists of mean and standard deviation values
                for each channel of the image.
        """
        sum_channels = torch.zeros(3)
        sum_sq_channels = torch.zeros(3)
        total_pixels = 0

        for sample in self:
            if self.image_mode in ['rgb', 'rgb_blend']:
                image_tensor = sample[0]
                if image_tensor.dtype == torch.uint8:
                    image_tensor = image_tensor.to(torch.float32) / 255.0
            elif self.image_mode == 'rgbh':
                image_tensor = sample[0][:3]
                image_tensor = image_tensor / 255.0
            else:
                raise ValueError(f"Unsupported image mode: {self.image_mode}")

            sum_channels += image_tensor.view(3, -1).sum(dim=1)
            sum_sq_channels += (image_tensor.view(3, -1) ** 2).sum(dim=1)
            total_pixels += image_tensor.size(1) * image_tensor.size(2)

        mean_channels = sum_channels / total_pixels
        std_channels = torch.sqrt((sum_sq_channels / total_pixels) - (mean_channels ** 2))

        return mean_channels.tolist(), std_channels.tolist()

    def calculate_mean_std_height(self) -> Tuple[float, float]:
        """
           Calculate the mean and standard deviation of the height channel across all images in the dataset.

           This method assumes that the dataset contains RGBH images (images with a height channel as the last channel).

           Returns:
               Tuple[float, float]: A tuple containing the mean and standard deviation of the height channel values.

           Raises:
               AssertionError: If the image mode is not 'rgbh'. This method is only supported for rgbh images.
        """
        assert self.image_mode == 'rgbh', "This is only supported for rgbh images"
        sum_channels = 0
        sum_sq_channels = 0
        total_pixels = 0
        for image_tensor, target in self:
            height_channel = image_tensor[-1]
            sum_channels += height_channel.sum().item()
            sum_sq_channels += (height_channel ** 2).sum().item()
            total_pixels += height_channel.numel()

        mean = sum_channels / total_pixels
        std = ((sum_sq_channels / total_pixels) - (mean ** 2)) ** 0.5

        return mean, std

    # I use this for now, since I don't need the other augmentations.
    def _apply_albumentations_transforms_rgb(self, image: torch.Tensor,
                                             image_annotation_dict: Dict[str, Union[torch.Tensor, int]]) \
            -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, int]]]:
        """
        Apply Albumentations transforms to the input image and annotation dictionary. IMPORTANT: Spatial transformations,
        that need to interpolate (e.g. rotations that are no multiple of 90 degrees) or transformations that resize
        the dimensions of the image (e.g. crop), will not work with this method, due to the implementation
        of Albumentations. Below, is a method with a workaround that will make these augmentations
        work and a description of the problems that need to be addressed.

        Args:
            image (torch.Tensor): The input image tensor. Should be torch.uint8 or torch.float32
             and have a format of [C, H, W].
            image_annotation_dict (Dict[str, torch.Tensor]): A dictionary containing annotation tensors including masks,
                bounding boxes, labels, and image ID.

          Notes:
                - Depending on the augmentation the datatype of the returned image tensor is either torch.uint8 or torch.float32.
                - The dictionary contains the following keys: 'masks', 'bounding_boxes', 'labels', 'image_id'.
                - The entry for the key 'masks' is a torch.uint8 [N, H, W] tensor.
                - The entry for the key 'bounding_boxes' is a torch.float32 [N,4] tensor in 'xyxy' format.
                - The entry for the key 'labels' is a torch.int64 scalar [N] tensor.
                - N is the number of masks on the image and also specifies the number of corresponding bounding boxes and labels.
                - The entry for the key 'image_id' gives back an integer, which is the filename of the image.

        Returns:
            Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, int]]]: A tuple containing the transformed image
                tensor and the transformed annotation dictionary.
            """
        assert self.image_mode != 'rgbh', "This is only supported for rgb images"
        transformed = self.transforms(
            image=image.numpy(force=True).transpose((1, 2, 0)),
            mask=image_annotation_dict['masks'].numpy(force=True).transpose((1, 2, 0)),
            bboxes=image_annotation_dict['boxes'].numpy(force=True),
            bbox_classes=image_annotation_dict['labels'].numpy(force=True)
        )

        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]
        bboxes_tensor = convert_bbox_list_to_n4_tensor(transformed["bboxes"], dtype=torch.float32)
        labels_tensor = convert_label_list_to_n_tensor(transformed["bbox_classes"], dtype=torch.int64)
        self._validate_input_mask_rcnn(image_tensor, mask_tensor, bboxes_tensor, labels_tensor)

        return image_tensor, {'masks': mask_tensor, 'boxes': bboxes_tensor, 'labels': labels_tensor,
                              'image_id': image_annotation_dict['image_id']}

    # I currently don't need this workaround, since I don't use geometric transformations
    # that need interpolation of pixel values (i.e. rotation).
    # Could be nice for documentation though (why do you not use rotation for instance).
    def _apply_albumentations_transform_with_workaround(self, image: torch.Tensor,
                                                        image_annotation_dict: Dict[str, Union[torch.Tensor, int]]) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
            Apply transformations to an image and its associated annotations, with a workaround.
            This workaround prevents two inconsistencies from Albumentations. 1) Albumentations returns empty masks,
            but not empty bounding boxes and labels, if they are no longer in the image.
            2) Albumentations cannot deal with masks of shape [0, H, W], which happens when no instance
            is on image. In this case, Albumentations still tries to apply the transformation,
            which leads to a segmentation fault.

            Args:
                image (torch.Tensor[torch.uint8]): Input image [C, H, W] tensor (either torch.uint8 or torch.float32).
                image_annotation_dict (Dict[str, Union[torch.Tensor, int]): A dictionary containing image annotations
                    including masks, boxes, and labels. It must have keys 'masks', 'boxes', and 'labels' each mapping
                    to a tensor. The mask tensor must be a torch.uint8 [N, H, W] tensor, boxes [N,4]
                    torch.float32 tensor, and labels [N] torch.int64 tensor.

            Returns:
                Tuple[torch.Tensor, Dict[str, torch.Tensor]]: A tuple containing the transformed image tensor
                and a dictionary with transformed annotations including masks, bounding boxes, and labels.
                 The dictionary has keys 'masks', 'boxes', and 'labels' each mapping to a tensor.

            Raises:
                ValueError: If the mask tensor shape is incompatible.
            """
        original_mask_is_empty = image_annotation_dict['masks'].numel() == 0  # mask of shape (0, H, W)
        mask = image_annotation_dict['masks']

        # prevents memory issues with empty masks, by filling it with 0
        if original_mask_is_empty:  # fill mask with only 0 (since 0 means no instance and 1 means instance)
            mask = torch.zeros(1, mask.shape[1], mask.shape[2], dtype=torch.uint8)  # mask is now (1, H, W)

        transformed = self.transforms(
            image=image.numpy(force=True).transpose((1, 2, 0)),
            mask=mask.numpy(force=True).transpose((1, 2, 0)),
            bboxes=image_annotation_dict['boxes'].numpy(force=True),
            bbox_classes=image_annotation_dict['labels'].numpy(force=True)
        )

        transformed_mask = transformed['mask']
        # copy original labels (due to the deletion of boxes and labels but not empty masks)
        labels = image_annotation_dict['labels']

        if original_mask_is_empty:
            # if the original mask was empty, mask after transformation should be as well
            transformed_mask = torch.empty((0, transformed_mask.shape[1], transformed_mask.shape[2]), dtype=torch.uint8)
        else:
            # check if mask is empty after operation and remove empty masks and labels
            for i in range(transformed_mask.shape[0]):
                if torch.all(transformed_mask[0] == 0):
                    transformed_mask = transformed_mask[1:]
                    labels = labels[1:]

        # Generate bounding boxes based on new mask - Important since operations
        # like rotations alter boxes
        bboxes_xyxy = torchvision.ops.masks_to_boxes(transformed_mask)
        image_tensor = transformed["image"]
        self._validate_input_mask_rcnn(image_tensor, transformed_mask, bboxes_xyxy, labels)

        return image_tensor, {'masks': transformed_mask, 'boxes': bboxes_xyxy, 'labels': labels,
                              'image_id': image_annotation_dict['image_id']}

    def _validate_input_mask_rcnn(self, image_tensor: torch.Tensor, mask_tensor: torch.Tensor,
                                  bbox_tensor: torch.Tensor, label_tensor: torch.Tensor):
        """
        Validates the input tensors for Mask R-CNN to ensure they meet the required format and shape.

        Parameters:
        image_tensor (torch.Tensor): The tensor representing the input image. Expected to be of shape
                                     [C, H, W] where C can be 1 (grayscale) or 3 (RGB). The data type
                                     should be either torch.uint8 or torch.float32.
        mask_tensor (torch.Tensor): The tensor representing the instance masks. Expected to be of shape
                                    [N, H, W] where N is the number of instances, H is the height, and W
                                    is the width of the mask. The data type should be torch.uint8.
        bbox_tensor (torch.Tensor): The tensor representing the bounding boxes. Expected to be of shape
                                    [N, 4] where N is the number of bounding boxes, and 4 corresponds
                                    to the coordinates of each bounding box (x1, y1, x2, y2). The data
                                    type should be torch.float32.
        label_tensor (torch.Tensor): The tensor representing the labels of the instances. Expected to
                                     be of shape [N] where N is the number of instances. The data type
                                     should be torch.int64.

        Raises:
        AssertionError: If any of the following conditions are not met:
            - Bounding box tensor is not of shape [N, 4].
            - Number of bounding boxes, masks, and labels do not match.
            - Mask tensor is not of type torch.uint8.
            - Bounding box tensor is not of type torch.float32.
            - Image tensor is not of type torch.uint8 or torch.float32.
            - Label tensor is not of type torch.int64.
            - Mask dimensions do not match the image dimensions.
            - Image tensor is not in the format [C, H, W] with C=1 (grayscale) or C=3 (RGB).

        """
        assert bbox_tensor.shape[1] == 4, (""
                                           "Bounding box tensor is not [N, 4], where N is number of boxes on image")
        assert bbox_tensor.shape[0] == mask_tensor.shape[0] == label_tensor.shape[0], ("Number of bounding boxes and "
                                                                                       "masks do not match."
                                                                                       "Mask must be [N, H, W] and "
                                                                                       "bounding boxes must be ["
                                                                                       "N, 4]")
        assert mask_tensor.dtype == torch.uint8, "Masks tensor should be in uint8 format."
        assert bbox_tensor.dtype == torch.float32, "Bounding boxes should be in float32 format."
        assert image_tensor.dtype == torch.uint8 or image_tensor.dtype == torch.float32, \
            "Image tensor should be torch.float32 or torch.uint8"
        assert label_tensor.dtype == torch.int64, "Labels tensor should be in torch.int64 format."

        assert self.image_height == mask_tensor.shape[1] and self.image_width == mask_tensor.shape[2], \
            "Mask must be of shape [N, H, W], where N is the number of instances on the image"
        assert ((image_tensor.shape[0] == 1 or image_tensor.shape[0] == 3)
                and image_tensor.shape[1] == self.image_height and image_tensor.shape[2] == self.image_width), \
            "Image tensor must be in format [C, H, W] with C=1 (greyscale) or C=3 (RGB)"

    def _load_tensors_from_file(self, index):
        """
           Load tensors from files for a specific index. If the image mode is 'rgb' or 'rgb_blend', a tuple
           of (ortho_tensor, masks_tensor, bboxes_tensor) will be returned. If the mode is 'rgbh', a tuple of
            (ortho_tensor, dsm_tensor, masks_tensor, bboxes_tensor) will be returned.

           Args:
               index (int): Index of the tensors to load.
        """
        ortho_path = self.file_dictionary['orthophoto_tensor_patches_folder']
        masks_path = self.file_dictionary['mask_tensor_patches_folder']
        bboxes_path = self.file_dictionary['bbox_tensor_patches_folder']

        ortho_tensor = torch.load(os.path.join(ortho_path, f"{index}.pt"))
        masks_tensor = torch.load(os.path.join(masks_path, f"{index}.pt"))
        bboxes_tensor = torch.load(os.path.join(bboxes_path, f"{index}.pt"))

        if self.image_mode == 'rgb' or self.image_mode == 'rgb_blend':
            return ortho_tensor, masks_tensor, bboxes_tensor
        else:
            dsm_path = self.file_dictionary['dsm_tensor_patches_folder']
            dsm_tensor = torch.load(os.path.join(dsm_path, f"{index}.pt"))
            return ortho_tensor, dsm_tensor, masks_tensor, bboxes_tensor


if __name__ == '__main__':
    print("")
    # test = StockpileDataset("historic/val_no_over", "", transforms=None)
