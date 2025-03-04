import json
import os
from typing import List, Tuple

import numpy as np
import rasterio
import torch
import torchvision
from rasterio._base import Affine
from rasterio.features import geometry_mask
from shapely.geometry import shape
from shapely.geometry import MultiPolygon
from merge_and_combine import split_tiff_image_into_tiff_patches, \
    convert_temporary_tiff_patches_to_torch_tensors
from raster_processing import create_slope_raster_from_dem, create_blending_dem_ortho
from utils import normalise_image_tensor_data_0_to_1


class StockpileDatasetBuilderConfig:
    """
        Configuration settings for building a StockpileDataset.

        This class contains various configuration settings required for building a StockpileDataset. It includes
        parameters such as label mapping, image dimensions, patch overlap, padding values, and file paths.

        Attributes:
            LABEL_MAP (dict): A dictionary mapping class indices to class names.
            NUM_CLASSES (int): The number of classes in the dataset.
            IMAGE_WIDTH (int): The width of image patches.
            IMAGE_HEIGHT (int): The height of image patches.
            PATCH_OVERLAP (int): The overlap between adjacent patches in both directions.
            PADDING_VALUE_ORTHO (int or float): The padding value for orthophotos.
            PADDING_VALUE_DSM (float): The padding value for Digital Surface Models (DSM).
            REMOVE_PADDED_PIXELS_THRESHOLD (float): The threshold for removing images with padded pixels
                from the dataset.
            _REMOVE_MASK_THRESHOLD_PERCENT (float): The percentage of minimum mask area to keep the
            mask and the corresponding bounding box and label annotations.
            REMOVE_MASK_THRESHOLD_PIXELS (int): The minimum number of pixels to keep the mask and the corresponding
                bounding box and label annotations.
            KEPT_NUMBER_OF_BANDS_ORTHOPHOTO (int or None): The number of bands to keep in orthophotos.
            _INPUT_FOLDER_ARTIFACTS (str): The folder containing input data artifacts.
            INPUT_PATH_ORTHOPHOTO_TRAIN (str): The path to the orthophoto training dataset.
            INPUT_PATH_DSM_TRAIN (str): The path to the DSM training dataset.
            INPUT_PATH_ORTHOPHOTO_VAL (str): The path to the orthophoto validation dataset.
            INPUT_PATH_DSM_VAL (str): The path to the DSM validation dataset.
            INPUT_PATH_ORTHOPHOTO_TEST (str): The path to the orthophoto testing dataset.
            INPUT_PATH_DSM_TEST (str): The path to the DSM testing dataset.
            INPUT_FOLDER_POLYGONS (str): The folder containing polygon data.
            OUTPUT_FOLDER_TEMPORARY_TIFF_PATCHES (str): The folder for storing temporary TIFF patches.
            OUTPUT_FOLDER_ARTIFACTS (str): The folder for storing output dataset artifacts.
        """
    # must be changed for implementation with more than 2 labels
    # datatype must be changed as well and adjusted in masks
    LABEL_MAP = {
        0: "Background",
        1: "Sand Stockpile"
    }
    NUM_CLASSES = len(LABEL_MAP)
    IMAGE_WIDTH = 1024  # determines patch width
    IMAGE_HEIGHT = 1024  # determines patch height
    PATCH_OVERLAP = 512  # set overlap of patches
    PADDING_VALUE_ORTHO = 0  # padding value for orthophoto is black
    PADDING_VALUE_DSM = -9999.0  # padding value for DSM correspond to no_data value for dsm
    # if number of padded pixels / number of pixels
    # is higher than threshold, image will be removed
    REMOVE_PADDED_PIXELS_THRESHOLD = 0.95

    # adjust percentage for minimum area of mask (and corresponding bboxes + labels)
    _REMOVE_MASK_THRESHOLD_PERCENT = 0.025  # mask must fill at least x % of the image,
    # otherwise mask (and corresponding bboxes + labels) are removed and the area is considered "noise"
    REMOVE_MASK_THRESHOLD_PIXELS = int(round(_REMOVE_MASK_THRESHOLD_PERCENT * (IMAGE_WIDTH * IMAGE_HEIGHT)))

    # do not keep the alpha channel of the orthophoto (no use). Set this to None to keep all bands
    KEPT_NUMBER_OF_BANDS_ORTHOPHOTO = 3

    # factor for rgb 'blend'
    ALPHA_BLEND = 0.6

    # File and Paths
    _INPUT_FOLDER_ARTIFACTS = "resources/Split_Ortho_DSM_DTM"
    INPUT_PATH_ORTHOPHOTO_TRAIN = os.path.join(_INPUT_FOLDER_ARTIFACTS, "train_ortho.tif")
    INPUT_PATH_DSM_TRAIN = os.path.join(_INPUT_FOLDER_ARTIFACTS, "train_dsm.tif")
    INPUT_PATH_ORTHOPHOTO_VAL = os.path.join(_INPUT_FOLDER_ARTIFACTS, "val_ortho.tif")
    INPUT_PATH_DSM_VAL = os.path.join(_INPUT_FOLDER_ARTIFACTS, "val_dsm.tif")
    INPUT_PATH_ORTHOPHOTO_TEST = os.path.join(_INPUT_FOLDER_ARTIFACTS, "test_ortho.tif")
    INPUT_PATH_DSM_TEST = os.path.join(_INPUT_FOLDER_ARTIFACTS, "test_dsm.tif")
    INPUT_FOLDER_POLYGONS = "resources/Polygons_Stockpile"
    OUTPUT_FOLDER_TEMPORARY_TIFF_PATCHES = "temp_patches"  # needed for mask generation
    OUTPUT_FOLDER_TEMPORARY_SLOPE_RASTER = "temp_slope_raster"
    OUTPUT_FOLDER_ARTIFACTS = "dataset"


class StockpileDatasetBuilder:
    """
    A class to build a dataset for stockpile analysis using orthophoto (RGB) and
    optionally DSM (Digital Surface Model) images.

    Attributes:
        has_been_built (bool or None): Externally set if dataset has been built. Defaults to None, but will be set to `False` in .build(), which triggers a rebuild.
        mode (str): The mode of the dataset builder, which can be "train", "val", or "test".
        image_mode (str): The image mode of the dataset. Can be 'rgb', 'rgb_blend', or 'rgbh'.
    """

    def __init__(self, dataset_mode: str, image_mode: str) -> None:
        """
        Initializes the StockpileDatasetBuilder instance.

        :param dataset_mode: Specifies the mode of the dataset builder, which can be "train", "val", or "test".
        :param image_mode: The image mode of the dataset. Can be 'rgb', 'rgb_blend', or 'rgbh'.
        """
        # Externally set if dataset has been built
        self.has_been_built = None

        # Set before building
        self.mode = dataset_mode
        self.image_mode = image_mode

        self.alpha_blend = None
        # Conditional on image mode
        if self.image_mode == 'rgb_blend':
            self.alpha_blend = StockpileDatasetBuilderConfig.ALPHA_BLEND

        # Set Label Map
        self.label_map = StockpileDatasetBuilderConfig.LABEL_MAP

        # create paths and folders
        self._create_input_and_output_paths(self.mode, self.image_mode)

        # set values for patch creation
        self._kept_number_of_bands_orthophoto = StockpileDatasetBuilderConfig.KEPT_NUMBER_OF_BANDS_ORTHOPHOTO
        self._patch_size = (StockpileDatasetBuilderConfig.IMAGE_WIDTH, StockpileDatasetBuilderConfig.IMAGE_HEIGHT)
        self._patch_overlap = StockpileDatasetBuilderConfig.PATCH_OVERLAP
        self._padding_value_orthophoto = StockpileDatasetBuilderConfig.PADDING_VALUE_ORTHO
        self._padding_value_dsm = StockpileDatasetBuilderConfig.PADDING_VALUE_DSM
        self._remove_padded_pixels_threshold = StockpileDatasetBuilderConfig.REMOVE_PADDED_PIXELS_THRESHOLD
        self._remove_mask_threshold = StockpileDatasetBuilderConfig.REMOVE_MASK_THRESHOLD_PIXELS
        self._image_width = StockpileDatasetBuilderConfig.IMAGE_WIDTH
        self._image_height = StockpileDatasetBuilderConfig.IMAGE_HEIGHT

    def set_has_been_built(self, has_been_built: bool) -> 'StockpileDatasetBuilder':
        """
        Sets the `has_been_built` attribute of the StockpileDatasetBuilder instance.
        If it is set to `True`, the dataset will not be rebuilt on disc and will be loaded instead,
        otherwise the dataset will be built on disc.

        :param has_been_built: Indicates whether the dataset has been built before.
        :return: The instance itself for chaining method calls.
        """
        self.has_been_built = has_been_built
        return self

    def _create_input_and_output_paths(self, mode: str, image_mode: str) -> None:
        """
         Creates necessary input and output paths based on the mode and data usage.
        :param mode: The mode of the dataset builder, which can be "train", "val", or "test".
        :param image_mode: The image mode of the dataset. Can be 'rgb', 'rgb_blend', or 'rgbh'.
        """
        if mode == "train":
            self._input_path_orthophoto = StockpileDatasetBuilderConfig.INPUT_PATH_ORTHOPHOTO_TRAIN
            self._input_path_dsm = StockpileDatasetBuilderConfig.INPUT_PATH_DSM_TRAIN
        elif mode == "val":
            self._input_path_orthophoto = StockpileDatasetBuilderConfig.INPUT_PATH_ORTHOPHOTO_VAL
            self._input_path_dsm = StockpileDatasetBuilderConfig.INPUT_PATH_DSM_VAL
        else:
            self._input_path_orthophoto = StockpileDatasetBuilderConfig.INPUT_PATH_ORTHOPHOTO_TEST
            self._input_path_dsm = StockpileDatasetBuilderConfig.INPUT_PATH_DSM_TEST

        self._input_folder_polygons = StockpileDatasetBuilderConfig.INPUT_FOLDER_POLYGONS

        self._output_folder_orthophoto_patches = os.path.join(StockpileDatasetBuilderConfig.OUTPUT_FOLDER_ARTIFACTS,
                                                              f"{mode}/ortho_patches")
        if self.image_mode == 'rgb_blend' or image_mode == 'rgbh':
            self._output_folder_dsm_patches = os.path.join(StockpileDatasetBuilderConfig.OUTPUT_FOLDER_ARTIFACTS,
                                                           f"{mode}/dsm_patches")
            if self.image_mode == 'rgb_blend':
                self._temporary_slope_raster = StockpileDatasetBuilderConfig.OUTPUT_FOLDER_TEMPORARY_SLOPE_RASTER
        elif self.image_mode == 'rgb':
            self._output_folder_dsm_patches = None

        self._output_folder_bounding_boxes = os.path.join(StockpileDatasetBuilderConfig.OUTPUT_FOLDER_ARTIFACTS,
                                                          f"{mode}/bbox_patches_xyxy")

        self._output_folder_segmentation_masks = os.path.join(StockpileDatasetBuilderConfig.OUTPUT_FOLDER_ARTIFACTS,
                                                              f"{mode}/seg_masks_patches")

        self._output_folder_temporary_tiff_patches = StockpileDatasetBuilderConfig.OUTPUT_FOLDER_TEMPORARY_TIFF_PATCHES

        # Create Output folders if they don't exist
        os.makedirs(self._output_folder_orthophoto_patches, exist_ok=True)
        if self.image_mode == 'rgbh':
            os.makedirs(self._output_folder_dsm_patches, exist_ok=True)
        os.makedirs(self._output_folder_bounding_boxes, exist_ok=True)
        os.makedirs(self._output_folder_segmentation_masks, exist_ok=True)

    def _create_dataset(self) -> None:
        """
            Create the dataset. Filenames are handled internally (0-indexed incrementing) to maintain
            consistency between the created image tensors, mask tensors and bounding box tensors.
        """
        # Orthophoto patches
        split_tiff_image_into_tiff_patches(path_tiff_image=self._input_path_orthophoto,
                                           location_folder=self._output_folder_temporary_tiff_patches,
                                           fill_value_padding=self._padding_value_orthophoto,
                                           patch=self._patch_size,
                                           kept_number_of_bands=self._kept_number_of_bands_orthophoto,
                                           overlap_in_px=self._patch_overlap,
                                           remove_padded_pixels_threshold=self._remove_padded_pixels_threshold)
        print("Created orthophoto patches successfully")

        # Load Polygons
        multipolygons = self._load_list_of_qgis_multipolygons()
        print("Loaded multipolygons successfully")

        # Load image data and transform of temporary tiff patches as np array
        tiff_patch_list, transform_list = self._extract_patches_and_transform()
        print("Loaded image data and transform of temporary tiff patches as np array successfully")

        # Create mask and bounding box tensors
        self._create_mask_tensors(multipolygons=multipolygons, tiff_patches_list=tiff_patch_list,
                                  transform_list=transform_list)
        self._create_bounding_box_tensors()
        print("Created mask and bounding box tensors successfully")

        # Save the Tiff patches as tensors and delete temporary file patches
        ortho_tensors = convert_temporary_tiff_patches_to_torch_tensors(
            path_folder_tiff_patches=self._output_folder_temporary_tiff_patches,
            path_result_folder=self._output_folder_orthophoto_patches)
        print("Saved tiff patches as tensors and deleted temporary files successfully")

        if self.image_mode == 'rgb_blend':
            # Create temporary slope raster
            os.makedirs(self._temporary_slope_raster, exist_ok=True)
            temporary_raster_filename = "temp.tif"
            slope_raster_path = os.path.join(self._temporary_slope_raster, temporary_raster_filename)
            create_slope_raster_from_dem(self._input_path_dsm, slope_raster_path)

            # get min, max, and no_data values for normalisation:
            min_height, max_height, no_data_height = self._get_min_max_no_data_height(slope_raster_path)

            # create temporary patches
            split_tiff_image_into_tiff_patches(path_tiff_image=slope_raster_path,
                                               location_folder=self._output_folder_temporary_tiff_patches,
                                               fill_value_padding=self._padding_value_dsm,
                                               patch=self._patch_size,
                                               overlap_in_px=self._patch_overlap,
                                               remove_padded_pixels_threshold=self._remove_padded_pixels_threshold)

            os.makedirs(self._output_folder_dsm_patches, exist_ok=True)

            # create tensors from patches
            height_tensors = convert_temporary_tiff_patches_to_torch_tensors(
                path_folder_tiff_patches=self._output_folder_temporary_tiff_patches,
                path_result_folder=self._output_folder_dsm_patches)

            # Save blended tensors (and overwrite ortho tensors)
            self._create_blended_tensors(ortho_tensors, height_tensors, min_height, max_height, no_data_height)

            # delete temporary files
            os.remove(slope_raster_path)
            os.rmdir(self._temporary_slope_raster)
            for item in os.listdir(self._output_folder_dsm_patches):
                os.remove(os.path.join(self._output_folder_dsm_patches, item))
            os.rmdir(self._output_folder_dsm_patches)

            print("Created Temporary Blend patches successfully")
        elif self.image_mode == 'rgbh':
            split_tiff_image_into_tiff_patches(path_tiff_image=self._input_path_dsm,
                                               location_folder=self._output_folder_temporary_tiff_patches,
                                               fill_value_padding=self._padding_value_dsm,
                                               patch=self._patch_size,
                                               overlap_in_px=self._patch_overlap,
                                               remove_padded_pixels_threshold=self._remove_padded_pixels_threshold)

            convert_temporary_tiff_patches_to_torch_tensors(
                path_folder_tiff_patches=self._output_folder_temporary_tiff_patches,
                path_result_folder=self._output_folder_dsm_patches, normalize=True,
                no_data_value=self._padding_value_dsm)

        # Delete Temporary Folder if empty
        if not os.listdir(self._output_folder_temporary_tiff_patches):
            os.rmdir(self._output_folder_temporary_tiff_patches)
            print("Temporary tiff patches folder deleted successfully")

    def _create_mask_tensors(self, multipolygons: List[MultiPolygon], tiff_patches_list: List[np.ndarray],
                             transform_list: List[Affine]) \
            -> None:
        """
        Creates mask tensors based on the given multipolygons, TIFF patches, and transformation matrices.
        The mask tensors are torch.uint8 and in [N, H, W] shape, where N is the number of detected masks for
        a given patch.

        :param multipolygons: A list of MultiPolygon. These are the segmentation masks.
        :param tiff_patches_list: A list of numpy arrays representing TIFF patches. These are the image patches.
        :param transform_list: A list of Affine transformation matrices. These are the affine transformations
        for the image patches.
        :raises AssertionError: If the length of `tiff_patches_list` does not match the length of `transform_list`.
        """
        assert len(tiff_patches_list) == len(transform_list), \
            "There must be an equal number of patches and corresponding transforms"
        for idx, patch in enumerate(tiff_patches_list):
            instance_masks = []
            for multipolygon in multipolygons:
                # each multipolygon can have multiple polygons (because of holes)
                resulting_instance_mask = np.zeros(patch[0].shape, dtype=bool)  # 0 is background, 1 is an instance
                for polygon in multipolygon.geoms:
                    polygon_mask = geometry_mask([polygon], out_shape=patch[0].shape,
                                                 transform=transform_list[idx], invert=True)
                    resulting_instance_mask = resulting_instance_mask | polygon_mask
                # remove polygon instances on padding
                area_without_padding_mask = ~ ((patch[0] == self._padding_value_orthophoto) &
                                               (patch[1] == self._padding_value_orthophoto) &
                                               (patch[2] == self._padding_value_orthophoto))
                resulting_instance_mask = resulting_instance_mask & area_without_padding_mask
                if not np.any(resulting_instance_mask):
                    continue  # instance not on image
                instance_masks.append(resulting_instance_mask)

            if len(instance_masks) == 0:  # case where no polygon is displayed on image
                empty_tensor = torch.empty((0,) + patch[0].shape, dtype=torch.uint8)
                torch.save(empty_tensor, os.path.join(self._output_folder_segmentation_masks, f"{idx}.pt"))
            else:
                torch.save(torch.tensor(np.stack(instance_masks), dtype=torch.uint8),
                           os.path.join(self._output_folder_segmentation_masks, f"{idx}.pt"))

    def _create_bounding_box_tensors(self) -> None:
        """
        Creates bounding box tensors in 'xyxy' format from segmentation mask tensors saved in the output folder.
        Filenames are determined internally (0-indexed incrementing) to maintain consistency between the files.
        The bounding box tensors are torch.float32 and have a shape of [N, 4],
        where N is the number of masks on the image patch, which are used to create the bounding box tensors.

        :raises AssertionError: If there are no segmentation mask tensors found in the output folder.
        """
        mask_filepaths = [os.path.join(self._output_folder_segmentation_masks, f) for
                          f in os.listdir(self._output_folder_segmentation_masks) if
                          f.endswith('.pt')]
        assert len(mask_filepaths) > 0, "No masks in folder {}".format(self._output_folder_segmentation_masks)
        for mask_filepath in mask_filepaths:
            mask_tensor = torch.load(mask_filepath)
            bboxes_xyxy = torchvision.ops.masks_to_boxes(mask_tensor)
            torch.save(bboxes_xyxy,
                       os.path.join(self._output_folder_bounding_boxes,
                                    f"{os.path.splitext(os.path.basename(mask_filepath))[0]}.pt"))

    def _get_tiff_patches_filepath(self) -> List[str]:
        """
        Retrieves file paths of TIFF patches from the output folder. Filenames are determined internally
        for consistency.
        """
        tiff_fns = [os.path.join(self._output_folder_temporary_tiff_patches, f) for
                    f in os.listdir(self._output_folder_temporary_tiff_patches) if
                    f.endswith('.tif')]
        tiff_fns = sorted(tiff_fns, key=lambda x: int(os.path.basename(os.path.splitext(x)[0])))
        assert len(tiff_fns) > 0, "No tiffs found in folder {}".format(self._output_folder_temporary_tiff_patches)
        return tiff_fns

    def _extract_patches_and_transform(self) -> Tuple[List[np.ndarray], List[Affine]]:
        """
        Extracts patches from TIFF files and their corresponding transformation matrices.
        :return: A tuple containing lists of extracted patches and their transformation matrices.
        """
        tiff_patches_filepath = self._get_tiff_patches_filepath()
        tiff_patch_list = []
        transform_list = []
        for tiff_patch_filepath in tiff_patches_filepath:
            with rasterio.open(tiff_patch_filepath, 'r') as src:
                if self._kept_number_of_bands_orthophoto is None:
                    self._kept_number_of_bands_orthophoto = src.meta['count']
                img_data = src.read(indexes=list(range(1, self._kept_number_of_bands_orthophoto + 1)))
                tiff_patch_list.append(img_data)
                transform_list.append(src.transform)
        return tiff_patch_list, transform_list

    def _load_list_of_qgis_multipolygons(self) -> List[MultiPolygon]:
        """
       Load a list of QGIS multipolygons from GeoJSON files.

       :return: A list of loaded QGIS multipolygons.
       :raises AssertionError: If there are no GeoJSON files found in the input folder.
        """
        path_polygons = [file for file in os.listdir(self._input_folder_polygons) if file.endswith('.geojson')]
        assert len(path_polygons) > 0, "No polygons found in input folder. Only .geojson files are accepted."
        multipolygons = []
        for file in path_polygons:
            for multipolygon in self._load_multipolygons_from_qgis_geojson(os.path.join(self._input_folder_polygons,
                                                                                        file)):
                multipolygons.append(multipolygon)
        return multipolygons

    def _create_blended_tensors(self, ortho_tensors: torch.Tensor, height_tensors: torch.Tensor,
                                min_value: float, max_value: float, no_data_value: float):
        """
        Create blended tensors from ortho and height tensors, and save
        the resulting tensors to disk.

        Args:
            ortho_tensors (torch.Tensor): [N, C, H, W] orthophoto tensor, where N is the number of ortho tensors.
            height_tensors (torch.Tensor): [N, C, H, W] height tensor, where N is the number of height tensors.
            min_value (float): Minimum value for normalizing height tensors.
            max_value (float): Maximum value for normalizing height tensors.
            no_data_value (float): Value representing no data in height tensors.

        Saves:
            Blended tensors as torch tensors (.pt files) to `self._output_folder_orthophoto_patches`.
        """
        for i, (ortho_tensor, height_tensor) in enumerate(zip(ortho_tensors, height_tensors)):
            ortho_np = ortho_tensor.numpy(force=True).transpose((1, 2, 0))  # [ H , W , C]

            height_tensor_normalised = normalise_image_tensor_data_0_to_1(height_tensor, max_value,
                                                                          min_value, no_data_value)

            height_np_normalised = height_tensor_normalised.numpy(force=True).transpose((1, 2, 0))

            blended_image_uint8 = create_blending_dem_ortho(height_np_normalised, ortho_np, self.alpha_blend)

            result_tensor = torch.from_numpy(blended_image_uint8.transpose(2, 0, 1))

            # overwrite old tensors
            torch.save(result_tensor, os.path.join(self._output_folder_orthophoto_patches, f"{i}.pt"))

    def _get_min_max_no_data_height(self, path_slope_raster: str) -> Tuple[float, float, float]:
        """
            Retrieve the minimum, maximum, and no-data value from a slope raster.

            Args:
                path_slope_raster (str): Path to the slope raster file.

            Returns:
                tuple: A tuple containing the minimum, maximum, and no-data value.
        """
        with rasterio.open(path_slope_raster, "r") as dsm:
            dsm_np = dsm.read().transpose(1, 2, 0)  # [H, W, C]
            no_data_value = dsm.meta['nodata']
            masked_arr = np.ma.masked_values(dsm_np, no_data_value)
            min_image_val = masked_arr.min()
            max_image_val = masked_arr.max()
            return min_image_val, max_image_val, no_data_value

    def _check_for_equal_number_of_files(self) -> int:
        """
        Checks if there is an equal number of files in output folders and returns the filecount.

        :return: The count of files in the orthophoto patches folder.
        :raises AssertionError: If the number of files in the output folders is not consistent or is zero.
        """
        fns_dsm = None
        if self.image_mode == 'rgbh':
            fns_dsm = [os.path.join(self._output_folder_dsm_patches, f) for
                       f in os.listdir(self._output_folder_dsm_patches) if
                       f.endswith('.pt')]
        fns_orthophoto = [os.path.join(self._output_folder_orthophoto_patches, f) for
                          f in os.listdir(self._output_folder_orthophoto_patches) if
                          f.endswith('.pt')]
        fns_masks = [os.path.join(self._output_folder_segmentation_masks, f) for
                     f in os.listdir(self._output_folder_segmentation_masks) if
                     f.endswith('.pt')]
        fns_bboxes = [os.path.join(self._output_folder_bounding_boxes, f) for
                      f in os.listdir(self._output_folder_bounding_boxes) if
                      f.endswith('.pt')]
        if fns_dsm is not None:
            assert len(fns_dsm) == len(fns_orthophoto) == len(fns_masks) == len(fns_bboxes) > 0, \
                ("Number of files for DSM Patches, Orthophoto Patches, Masks, and "
                 "Bounding Boxes must be the same and bigger than 0.")
        else:
            assert len(fns_orthophoto) == len(fns_masks) == len(fns_bboxes) > 0, \
                ("Number of files for Orthophoto Patches, Masks, and "
                 "Bounding Boxes must be the same and bigger than 0.")
        return len(fns_orthophoto)

    @staticmethod
    def _load_multipolygons_from_qgis_geojson(path_geojson) -> List[MultiPolygon]:
        """
        Load multipolygons from a QGIS-generated GeoJSON file.
        :param path_geojson: The path to the QGIS-generated GeoJSON file.
        :return: A list of multipolygons extracted from the GeoJSON file.
        """
        with open(path_geojson, 'r') as poly_source:
            geojson_data = json.load(poly_source)
            resulting_multipolygons = []
            for i in range(len(geojson_data['features'])):
                resulting_multipolygons.append(shape(geojson_data['features'][i]['geometry']))

            return resulting_multipolygons

    def build(self):
        """
        Build the StockpileDataset instance based on the configuration. If you try to build a dataset
        with `has_been_built` set, the method verifies that it can find an equal number of files for the
        masks, bounding_boxes,  orthophoto images, and optionally height images. If that is not the case,
        an assertion error will be raised.

        :return: The constructed StockpileDataset instance.
        """
        from stockpile_dataset import StockpileDataset

        if self.has_been_built is None:
            self.has_been_built = False  # rebuild if not set

        file_dictionary = None
        if self.image_mode == 'rgb' or self.image_mode == 'rgb_blend':
            file_dictionary = {
                "orthophoto_tensor_patches_folder": self._output_folder_orthophoto_patches,
                "mask_tensor_patches_folder": self._output_folder_segmentation_masks,
                "bbox_tensor_patches_folder": self._output_folder_bounding_boxes
            }
        elif self.image_mode == 'rgbh':
            file_dictionary = {
                "orthophoto_tensor_patches_folder": self._output_folder_orthophoto_patches,
                "dsm_tensor_patches_folder": self._output_folder_dsm_patches,
                "mask_tensor_patches_folder": self._output_folder_segmentation_masks,
                "bbox_tensor_patches_folder": self._output_folder_bounding_boxes
            }

        if self.has_been_built:
            length = self._check_for_equal_number_of_files()
            return StockpileDataset(image_mode=self.image_mode, file_dictionary=file_dictionary, length=length,
                                    label_map=self.label_map, remove_mask_threshold=self._remove_mask_threshold,
                                    image_width=self._image_width, image_height=self._image_height)
        else:
            # only create temp folder if dataset has not been built
            os.makedirs(self._output_folder_temporary_tiff_patches, exist_ok=True)
            self._create_dataset()
            print("###### Dataset successfully built. ######")
            length = self._check_for_equal_number_of_files()
            print("###### Same number of files created ######")
            return StockpileDataset(image_mode=self.image_mode, file_dictionary=file_dictionary, length=length,
                                    label_map=self.label_map, remove_mask_threshold=self._remove_mask_threshold,
                                    image_width=self._image_width, image_height=self._image_height)
