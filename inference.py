import json
import os

import cv2
import numpy as np
import rasterio
import torch
from cjm_pytorch_utils.core import get_torch_device
from matplotlib import pyplot as plt
from shapely import MultiPolygon, MultiPoint, Point
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

from inference_utils import load_final_multipolygons_and_geojson, load_tensors_in_correct_order_on_cpu, \
    remove_small_box_in_large_box, \
    merge_patches_with_overlap_top_and_left, load_polygons_in_correct_order, create_height_difference_mask, \
    create_volume_polygons_with_dsm_and_dtm_values, create_volume_polygons_with_dsm_values, \
    check_correct_number_of_files_created, create_dtm_polygons, create_dsm_polygons_with_dtm_and_original_multipolygons, \
    save_polygon_to_disk, create_crs_polygon_from_contour, merge_patches_with_overlap_top_or_left, \
    transform_epsg32632_geometry_to_wgs84, check_same_pixel_size_dsm_dtm_orthophoto
from merge_and_combine import split_tiff_image_into_tiff_patches, convert_temporary_tiff_patches_to_torch_tensors, \
    find_image_size_with_overlap
from raster_processing import create_slope_raster_from_dem, create_blend_rgb_ortho_with_dem_raster, \
    check_alignment_tif_files, align_dem_to_orthophoto, get_no_data_value
from utils import get_model_prediction_as_dict, load_mask_rcnn_on_device, parse_training_metadata_json
from visualise import visualise_merging_results
from rasterio.features import geometry_mask


class InferenceConfig:
    # Path Directories
    CURRENT_DIRECTORY = os.path.dirname(__file__)
    DIRECTORY_BASE = os.path.join(CURRENT_DIRECTORY, "inference")
    DIRECTORY_INPUTS = os.path.join(DIRECTORY_BASE, "inputs")
    DIRECTORY_OUTPUTS = os.path.join(DIRECTORY_BASE, "outputs")

    # Path Inputs
    MODEL_PATH = os.path.join(DIRECTORY_INPUTS, "inference_model/mask_rcnn_resnet50_fpn.pt")
    MODEL_METADATA_PATH = os.path.join(DIRECTORY_INPUTS, "inference_model/training_metadata_parameters")
    PATH_ORTHOPHOTO = os.path.join(DIRECTORY_INPUTS, "orthophoto.tif")
    PATH_DSM = os.path.join(DIRECTORY_INPUTS, "dsm.tif")
    PATH_DTM = os.path.join(DIRECTORY_INPUTS, "dtm.tif")

    # Path Outputs
    DIRECTORY_ORTHOPHOTO_TENSORS = os.path.join(DIRECTORY_OUTPUTS, "orthophoto_tensors")
    DIRECTORY_DSM_TENSORS = os.path.join(DIRECTORY_OUTPUTS, "dsm_tensors")
    DIRECTORY_DTM_TENSORS = os.path.join(DIRECTORY_OUTPUTS, "dtm_tensors")
    DIRECTORY_MODEL_BBOXES = os.path.join(DIRECTORY_OUTPUTS, "model_bbox")  # needed for pre processing
    DIRECTORY_MODEL_MASKS = os.path.join(DIRECTORY_OUTPUTS, "model_mask")  # needed
    # not needed in binary classification, still saved
    DIRECTORY_MODEL_LABELS = os.path.join(DIRECTORY_OUTPUTS, "model_label")
    DIRECTORY_MERGED_NP_ARRAY = os.path.join(DIRECTORY_OUTPUTS, "merged_predictions")
    PATH_MERGED_NP_ARRAY = os.path.join(DIRECTORY_MERGED_NP_ARRAY, "merged.npy")
    DIRECTORY_POLYGONS = os.path.join(DIRECTORY_OUTPUTS, "polygons")
    DIRECTORY_POLYGONS_CONVEX_HULL = os.path.join(DIRECTORY_OUTPUTS, "polygons_convex_hull")
    DIRECTORY_POLYGONS_DTM = os.path.join(DIRECTORY_OUTPUTS, "polygons_dtm")
    DIRECTORY_POLYGONS_DSM = os.path.join(DIRECTORY_OUTPUTS, "polygons_dsm")
    DIRECTORY_POLYGONS_FINAL = os.path.join(DIRECTORY_OUTPUTS, "polygons_result")
    DIRECTORY_LEAFLET_POLYGONS = os.path.join(DIRECTORY_OUTPUTS, "polygons_leaflet_wgs84")
    DIRECTORY_TEMPORARY_FILES = os.path.join(CURRENT_DIRECTORY, "temporary_files")  # will be deleted afterwards
    PATH_TEMPORARY_RASTER = os.path.join(DIRECTORY_TEMPORARY_FILES, "temp.tif")  # will be deleted afterwards

    # Internal Parameters (should not be determined by user)

    # Supported: 'rgb' , 'rgb_blend'. Currently no 'rgbh' support, since
    # 'rgb_blend' performs better in all aspects. Can be easily added later.
    MODE = parse_training_metadata_json(MODEL_METADATA_PATH, "mode")

    # if the DEM should be aligned to orthophoto (in case they have different dimensions)
    # if set to 'True' we try to align them, otherwise an exception will be thrown
    TRY_TO_ALIGN_DEM_TO_ORTHOPHOTO = True

    # error tolerance when checking equality of pixel size of dsm, dtm and orthophoto
    TOLERANCE_PIXEL_SIZE = 0.001

    # height and width should correspond to model
    PATCH_WIDTH = parse_training_metadata_json(MODEL_METADATA_PATH, "image_dimensions")[0]
    PATCH_HEIGHT = parse_training_metadata_json(MODEL_METADATA_PATH, "image_dimensions")[1]
    # can be freely chosen, but setting this too high will drastically increase
    # memory usage and computation time
    # if overlap is too high, results are worse
    # because the model is sensitive to FP
    OVERLAP = 256

    # Batch Size used for Inference (when doing predictions)
    # -> depends on available ram or vram
    BATCH_SIZE = 20
    THRESHOLD_SCORES = 0.5  # must be above this to consider prediction of model
    TRAINABLE_BACKBONE_LAYERS = parse_training_metadata_json(MODEL_METADATA_PATH, "trained_backbone_layers")
    NUMBER_OF_CLASSES = parse_training_metadata_json(MODEL_METADATA_PATH, "number_of_classes")
    IOU_THRESHOLD_MERGING_PATCHES = 0.5  # Threshold to merge instances in overlapping areas
    # if AREA_THRESHOLD_BOX_REMOVAL * area of a smaller box is inside a larger box,
    # remove the smaller box
    AREA_THRESHOLD_BOX_REMOVAL = 0.65

    # Threshold to find difference between dsm and dtm in m (to find conveyor belts)
    THRESHOLD_HEIGHT = 2

    ALPHA_BLEND = parse_training_metadata_json(MODEL_METADATA_PATH, "blend_alpha")
    DEVICE = torch.device(get_torch_device())  # for model predictions

    # Padding that is used for inference
    PADDING_ORTHO = 0
    PADDING_DTM = get_no_data_value(PATH_DTM)
    PADDING_DSM = get_no_data_value(PATH_DSM)

    # External Parameters (user should set these)

    # each spatially separated sand heap must have at
    # least this area in px to be considered as sand heap
    MIN_AREA_CONTOUR_PX = None

    # t/m^3
    SAND_DENSITY = None

    @classmethod
    def initialize(cls, min_area, sand_density):
        cls.MIN_AREA_CONTOUR_PX = min_area
        cls.SAND_DENSITY = sand_density


def try_to_align_rasters():
    """
      Check and attempt to align DSM and DTM rasters with an orthophoto.

      Raises:
          Exception: If DSM and DTM do not have the same pixel size or if DSM/DTM are not aligned with Orthophoto
           and alignment is not attempted.
    """
    same_px_size_dtm_dsm_orthophoto = check_same_pixel_size_dsm_dtm_orthophoto(InferenceConfig.PATH_DSM,
                                                                               InferenceConfig.PATH_DTM,
                                                                               InferenceConfig.PATH_ORTHOPHOTO,
                                                                               InferenceConfig.TOLERANCE_PIXEL_SIZE)

    if not same_px_size_dtm_dsm_orthophoto:
        # in this case, the volume should not be calculated as the data
        # does not allow correct calculations (needs to be handled manually)
        raise Exception("DSM, DTM, and Orthophoto do not have the same pixel size")

    aligned_dsm_ortho = check_alignment_tif_files(InferenceConfig.PATH_DSM, InferenceConfig.PATH_ORTHOPHOTO)
    aligned_dtm_ortho = check_alignment_tif_files(InferenceConfig.PATH_DTM, InferenceConfig.PATH_ORTHOPHOTO)

    if not aligned_dsm_ortho:
        if InferenceConfig.TRY_TO_ALIGN_DEM_TO_ORTHOPHOTO:
            align_dem_to_orthophoto(InferenceConfig.PATH_DSM, InferenceConfig.PATH_ORTHOPHOTO, InferenceConfig.PATH_DSM)
        else:
            raise Exception("DSM and Orthophoto are not aligned. Check if DSM and Orthophoto are in the same crs,"
                            "have the same dimensions, and have the same Affine Transformation")

    if not aligned_dtm_ortho:
        if InferenceConfig.TRY_TO_ALIGN_DEM_TO_ORTHOPHOTO:
            align_dem_to_orthophoto(InferenceConfig.PATH_DTM, InferenceConfig.PATH_ORTHOPHOTO, InferenceConfig.PATH_DTM)
        else:
            raise Exception("DTM and Orthophoto are not aligned. Check if DTM and Orthophoto are in the same crs,"
                            "have the same dimensions, and have the same Affine Transformation")


def create_tensor_patches():
    """
       Create tensor patches for orthophoto, DTM (Digital Terrain Model), and DSM (Digital Surface Model)
       from TIFF images and convert them to PyTorch tensors for inference.

       The paths and parameters for processing are obtained from the `InferenceConfig` class.

       Raises:
           OSError: If there's an error in file or directory operations.
       """
    os.makedirs(InferenceConfig.DIRECTORY_OUTPUTS, exist_ok=True)
    os.makedirs(InferenceConfig.DIRECTORY_ORTHOPHOTO_TENSORS, exist_ok=True)
    os.makedirs(InferenceConfig.DIRECTORY_DTM_TENSORS, exist_ok=True)
    os.makedirs(InferenceConfig.DIRECTORY_DSM_TENSORS, exist_ok=True)

    assert InferenceConfig.MODE == 'rgb' or InferenceConfig.MODE == 'rgb_blend', ("Currently only rgb and rgb_blend"
                                                                                  "are supported as inference mode.")

    if InferenceConfig.MODE == 'rgb_blend':
        os.makedirs(InferenceConfig.DIRECTORY_TEMPORARY_FILES, exist_ok=True)
        path_temp_raster = InferenceConfig.PATH_TEMPORARY_RASTER
        create_slope_raster_from_dem(path_dem=InferenceConfig.PATH_DSM,
                                     path_output=path_temp_raster)
        # Blended Orthophoto Raster
        create_blend_rgb_ortho_with_dem_raster(path_rgb_ortho=InferenceConfig.PATH_ORTHOPHOTO,
                                               path_dem_raster=path_temp_raster,
                                               path_output=path_temp_raster,
                                               blend_alpha_ortho=InferenceConfig.ALPHA_BLEND)
        # Blended Orthophoto Patches
        split_tiff_image_into_tiff_patches(path_tiff_image=path_temp_raster,
                                           location_folder=InferenceConfig.DIRECTORY_ORTHOPHOTO_TENSORS,
                                           fill_value_padding=InferenceConfig.PADDING_ORTHO,
                                           patch=(InferenceConfig.PATCH_WIDTH, InferenceConfig.PATCH_HEIGHT),
                                           kept_number_of_bands=3,
                                           overlap_in_px=InferenceConfig.OVERLAP)

        os.remove(path_temp_raster)
        os.rmdir(InferenceConfig.DIRECTORY_TEMPORARY_FILES)
    else:
        # Orthophoto Patches
        split_tiff_image_into_tiff_patches(path_tiff_image=InferenceConfig.PATH_ORTHOPHOTO,
                                           location_folder=InferenceConfig.DIRECTORY_ORTHOPHOTO_TENSORS,
                                           fill_value_padding=InferenceConfig.PADDING_ORTHO,
                                           patch=(InferenceConfig.PATCH_WIDTH, InferenceConfig.PATCH_HEIGHT),
                                           kept_number_of_bands=3,
                                           overlap_in_px=InferenceConfig.OVERLAP)

    # DSM PATCHES
    split_tiff_image_into_tiff_patches(path_tiff_image=InferenceConfig.PATH_DSM,
                                       location_folder=InferenceConfig.DIRECTORY_DSM_TENSORS,
                                       fill_value_padding=InferenceConfig.PADDING_DSM,
                                       patch=(InferenceConfig.PATCH_WIDTH, InferenceConfig.PATCH_HEIGHT),
                                       overlap_in_px=InferenceConfig.OVERLAP)

    # DTM PATCHES
    split_tiff_image_into_tiff_patches(path_tiff_image=InferenceConfig.PATH_DTM,
                                       location_folder=InferenceConfig.DIRECTORY_DTM_TENSORS,
                                       fill_value_padding=InferenceConfig.PADDING_DTM,
                                       patch=(InferenceConfig.PATCH_WIDTH, InferenceConfig.PATCH_HEIGHT),
                                       overlap_in_px=InferenceConfig.OVERLAP)

    # Convert Orthophoto Patches to Tensors
    convert_temporary_tiff_patches_to_torch_tensors(
        path_folder_tiff_patches=InferenceConfig.DIRECTORY_ORTHOPHOTO_TENSORS,
        path_result_folder=InferenceConfig.DIRECTORY_ORTHOPHOTO_TENSORS)

    # Convert DSM Patches to Tensors
    convert_temporary_tiff_patches_to_torch_tensors(
        path_folder_tiff_patches=InferenceConfig.DIRECTORY_DSM_TENSORS,
        path_result_folder=InferenceConfig.DIRECTORY_DSM_TENSORS)

    # Convert DTM Patches to Tensors
    convert_temporary_tiff_patches_to_torch_tensors(
        path_folder_tiff_patches=InferenceConfig.DIRECTORY_DTM_TENSORS,
        path_result_folder=InferenceConfig.DIRECTORY_DTM_TENSORS)

    # Small check
    check_correct_number_of_files_created(InferenceConfig.DIRECTORY_ORTHOPHOTO_TENSORS,
                                          InferenceConfig.DIRECTORY_DSM_TENSORS,
                                          InferenceConfig.DIRECTORY_DTM_TENSORS)


def create_model_predictions():
    """
        Generate model predictions for orthophoto tensor patches and save the results to disk.
        The model predictions are obtained in batches.

        The paths and parameters for processing are obtained from the `InferenceConfig` class.

        Raises:
            OSError: If there's an error in file or directory operations.
            RuntimeError: If there's an error during model prediction.
        """
    # Load Model
    model = load_mask_rcnn_on_device(path_model=InferenceConfig.MODEL_PATH, device=InferenceConfig.DEVICE,
                                     trainable_layers=InferenceConfig.TRAINABLE_BACKBONE_LAYERS,
                                     number_of_classes=InferenceConfig.NUMBER_OF_CLASSES)

    # Get Tensors from Directory
    tensors = load_tensors_in_correct_order_on_cpu(InferenceConfig.DIRECTORY_ORTHOPHOTO_TENSORS, stack=True)

    total_images = tensors.shape[0]

    # Create Directory for Predictions
    os.makedirs(InferenceConfig.DIRECTORY_MODEL_MASKS, exist_ok=True)
    os.makedirs(InferenceConfig.DIRECTORY_MODEL_LABELS, exist_ok=True)
    os.makedirs(InferenceConfig.DIRECTORY_MODEL_BBOXES, exist_ok=True)

    # Get Model Predictions in Batches and Save to Disk
    model.eval()
    for start in range(0, total_images, InferenceConfig.BATCH_SIZE):
        end = min(start + InferenceConfig.BATCH_SIZE, total_images)
        model_predictions = get_model_prediction_as_dict(model, tensors[start:end], InferenceConfig.THRESHOLD_SCORES,
                                                         device=InferenceConfig.DEVICE)
        for i, (bbox, label, masks) in enumerate(zip(model_predictions['bboxes'], model_predictions['labels'],
                                                     model_predictions['masks'])):
            torch.save(masks, os.path.join(InferenceConfig.DIRECTORY_MODEL_MASKS, f"{start + i}.pt"))
            torch.save(bbox, os.path.join(InferenceConfig.DIRECTORY_MODEL_BBOXES, f"{start + i}.pt"))
            torch.save(label, os.path.join(InferenceConfig.DIRECTORY_MODEL_LABELS, f"{start + i}.pt"))


def create_merged_predictions():
    """
        Merge model predictions from individual predicted patches into a single comprehensive result array.

        The paths and parameters for processing are obtained from the `InferenceConfig` class.

        Raises:
            OSError: If there's an error in file or directory operations.
        """
    # Get Image Width and Height of Orthophoto
    with rasterio.open(InferenceConfig.PATH_ORTHOPHOTO, 'r') as src:
        image_width = src.meta['width']
        image_height = src.meta['height']

    # Define resolution for resulting array
    patch_size_width = InferenceConfig.PATCH_WIDTH
    patch_size_height = InferenceConfig.PATCH_HEIGHT
    step_size_width = patch_size_width - InferenceConfig.OVERLAP
    step_size_height = patch_size_height - InferenceConfig.OVERLAP
    result_width, result_height = find_image_size_with_overlap(image_width=image_width, image_height=image_height,
                                                               step_size_width=step_size_width,
                                                               step_size_height=step_size_height,
                                                               patch_size_width=patch_size_width,
                                                               patch_size_height=patch_size_height)
    # Create empty array with correct resolution
    # IMPORTANT: Change from uint8 if you expect more than 255 (without background)
    # global instances
    result_array = np.empty((result_height, result_width), dtype=np.uint8)

    # Load the Tensors Predictions from the Model. Must add labels and handle them in multi class problem
    bbox_tensors = load_tensors_in_correct_order_on_cpu(InferenceConfig.DIRECTORY_MODEL_BBOXES, False)
    mask_tensors = load_tensors_in_correct_order_on_cpu(InferenceConfig.DIRECTORY_MODEL_MASKS, False)

    cur_img = 0
    cur_global_id = 1

    for y in range(0, result_height, step_size_height):
        for x in range(0, result_width, step_size_width):
            # do not take last step twice. happens in the overlap case because step size is smaller than patch size,
            # so remainder of last patch is taken twice
            if x + patch_size_width > result_width or y + patch_size_height > result_height:
                continue

            if mask_tensors[cur_img].numel() == 0:  # model has no predictions
                cur_img += 1
                continue

            # box and mask pre processing
            processed_bbox_tensor, processed_mask_tensor = remove_small_box_in_large_box(
                InferenceConfig.AREA_THRESHOLD_BOX_REMOVAL,
                bbox_tensors[cur_img], mask_tensors[cur_img])

            if x == 0 and y == 0:
                # first image, do not need to check overlap
                for mask in processed_mask_tensor:
                    instance_mask = mask == 1
                    result_array[y:y + patch_size_height, x:x + patch_size_width][instance_mask] = cur_global_id
                    cur_global_id += 1
                cur_img += 1
            elif y == 0:
                # first row, only check overlap to the left
                cur_global_id = merge_patches_with_overlap_top_or_left(result_array=result_array, x=x, y=y,
                                                                       patch_size_height=patch_size_height,
                                                                       patch_size_width=patch_size_width,
                                                                       overlap=InferenceConfig.OVERLAP,
                                                                       overlap_direction='left',
                                                                       processed_mask_tensor=processed_mask_tensor,
                                                                       iou_threshold=InferenceConfig.IOU_THRESHOLD_MERGING_PATCHES,
                                                                       cur_global_id=cur_global_id)
                cur_img += 1
            elif x == 0:
                # first column, only check for overlap to the top
                cur_global_id = merge_patches_with_overlap_top_or_left(result_array=result_array, x=x, y=y,
                                                                       patch_size_height=patch_size_height,
                                                                       patch_size_width=patch_size_width,
                                                                       overlap=InferenceConfig.OVERLAP,
                                                                       overlap_direction='top',
                                                                       processed_mask_tensor=processed_mask_tensor,
                                                                       iou_threshold=InferenceConfig.IOU_THRESHOLD_MERGING_PATCHES,
                                                                       cur_global_id=cur_global_id)
                cur_img += 1
            else:
                # all other patches, check overlap left and top
                cur_global_id = merge_patches_with_overlap_top_and_left(result_array=result_array, x=x, y=y,
                                                                        patch_size_height=patch_size_height,
                                                                        patch_size_width=patch_size_width,
                                                                        overlap=InferenceConfig.OVERLAP,
                                                                        processed_mask_tensor=processed_mask_tensor,
                                                                        iou_threshold=InferenceConfig.IOU_THRESHOLD_MERGING_PATCHES,
                                                                        cur_global_id=cur_global_id)

                cur_img += 1

    os.makedirs(InferenceConfig.DIRECTORY_MERGED_NP_ARRAY, exist_ok=True)
    np.save(InferenceConfig.PATH_MERGED_NP_ARRAY, result_array)


def create_qgis_multipolygons_with_merged_predictions():
    """
        Create and save the polygon masks (instances) from the merged prediction array as GeoJSON Polygon.


        The paths and parameters for processing are obtained from the `InferenceConfig` class.

        Raises:
            OSError: If there's an error in file or directory operations.
            RuntimeError: If there's an error during contour extraction or polygon creation.
        """
    with rasterio.open(InferenceConfig.PATH_ORTHOPHOTO) as src:
        crs_name = src.crs.data['init']
        transform = src.transform
        width = src.width
        height = src.height

    # only cut relevant part from the image
    merged_np_array = np.load(InferenceConfig.PATH_MERGED_NP_ARRAY)[0:height, 0:width]

    unique_numbers = np.unique(merged_np_array)
    cur_multipolygon = 0
    for number in unique_numbers:
        if number == 0:
            # Skip background
            continue

        # Find Contours for each instance on a binary image
        binary_mask = merged_np_array == number
        binary_instance_image = np.uint8(binary_mask * 255)
        contours, _ = cv2.findContours(binary_instance_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crs_polygons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # only consider sand heaps above min area for contours
            if area >= InferenceConfig.MIN_AREA_CONTOUR_PX:
                crs_polygon = create_crs_polygon_from_contour(contour, transform)
                crs_polygons.append(crs_polygon)

        multipolygon = MultiPolygon(crs_polygons)

        if not multipolygon.is_empty:
            # to save multipolygon for QGIS
            save_polygon_to_disk(InferenceConfig.DIRECTORY_POLYGONS, f"Multipolygon{cur_multipolygon}.geojson",
                                 multipolygon, crs_name)
            cur_multipolygon += 1


# purely for visualisation
def create_convex_hull_polygons():
    """
        Create the convex hull polygons from the predicted GeoJSON polygons and save them to disk.

        The paths and parameters for processing are obtained from the `InferenceConfig` class.

        Raises:
            AssertionError: If no polygon GeoJSON files are found in the specified directory.
            OSError: If there's an error in file or directory operations.
            ValueError: If there's an error during GeoJSON parsing or polygon processing.
        """
    path_polygons = [file for file in os.listdir(InferenceConfig.DIRECTORY_POLYGONS) if file.endswith('.geojson')]
    assert len(path_polygons) > 0, "No polygons found in input folder. Only .geojson files are accepted."

    for poly_path in path_polygons:
        with open(os.path.join(InferenceConfig.DIRECTORY_POLYGONS, poly_path), 'r') as poly_source:
            geojson_data = json.load(poly_source)

            convex_hull_polygons = []
            for polygon in shape(geojson_data['features'][0]['geometry']).geoms:
                convex_hull_polygons.append(polygon.convex_hull)

            if len(convex_hull_polygons) > 1:
                # Multipolygon with multiple polygons, find combined convex hull
                coords = [coord for polygon in convex_hull_polygons for coord in polygon.exterior.coords]
                joined_point_set = MultiPoint(coords)
                result_polygon = joined_point_set.convex_hull
            else:
                # only one polygon
                result_polygon = convex_hull_polygons[0]

            save_polygon_to_disk(InferenceConfig.DIRECTORY_POLYGONS_CONVEX_HULL,
                                 f"convex_hull_{geojson_data['name']}.geojson",
                                 result_polygon, geojson_data['crs']['properties']["name"])


# purely for visualisation
def visualise_diffed_dtm_dsm():
    """
        Visualize the height difference between DSM and DTM, highlighting areas above a certain height threshold.

        The paths and parameters for processing are obtained from the `InferenceConfig` class.

        Raises:
            OSError: If there's an error in file operations.
            RuntimeError: If there's an error during raster data processing.
        """
    with rasterio.open(InferenceConfig.PATH_DTM) as dtm_src, rasterio.open(InferenceConfig.PATH_DSM) as dsm_src:
        dtm_data = dtm_src.read(1)
        dsm_data = dsm_src.read(1)

        height_difference = dsm_data - dtm_data

        mask = height_difference > InferenceConfig.THRESHOLD_HEIGHT
        mask_image = np.zeros_like(dtm_data)
        mask_image[mask] = 127

        plt.imshow(mask_image, interpolation='none')
        plt.show()


# this function is purely for visualisation!
def create_dtm_dsm_polygons_for_visualisation():
    """
        Create and save DTM and DSM polygons to visualise the volume calculation.
        DTM polygons are the polygons where the final height is taken from the DTM (i.e.
        in areas where conveyor belts are above the sand heap). DSM polygons are the
        polygons where the height is taken from the DSM.

        The paths and parameters for processing are obtained from the `InferenceConfig` class.

        Raises:
            OSError: If there's an error in file operations.
            RuntimeError: If there's an error during raster data processing or polygon creation.
        """
    with rasterio.open(InferenceConfig.PATH_DTM) as dtm_src, rasterio.open(InferenceConfig.PATH_DSM) as dsm_src:
        dtm_data = dtm_src.read(1)
        dsm_data = dsm_src.read(1)
        transform = dtm_src.transform  # does not matter which, since they must have been aligned before
        crs_name = dtm_src.crs.data['init']

        height_mask = create_height_difference_mask(dsm_data, dtm_data, InferenceConfig.THRESHOLD_HEIGHT)

        # load convex hull and corresponding multipolygons
        original_multipolygons, convex_hull_polygons = load_polygons_in_correct_order(
            InferenceConfig.DIRECTORY_POLYGONS,
            InferenceConfig.DIRECTORY_POLYGONS_CONVEX_HULL)

        resulting_polygons_dtm = create_dtm_polygons(InferenceConfig.DIRECTORY_POLYGONS_DTM, convex_hull_polygons,
                                                     dtm_data, transform, crs_name, height_mask)

        create_dsm_polygons_with_dtm_and_original_multipolygons(InferenceConfig.DIRECTORY_POLYGONS_DSM,
                                                                resulting_polygons_dtm,
                                                                original_multipolygons, dtm_data, transform, crs_name,
                                                                InferenceConfig.MIN_AREA_CONTOUR_PX)


def create_polygons_with_volume():
    """
        Create and save polygons with volume calculations based on DTM and DSM data.

        The paths and parameters for processing are obtained from the `InferenceConfig` class.

        Raises:
            AssertionError: If the pixel sizes of DTM and DSM do not match.
            OSError: If there's an error in file operations.
            RuntimeError: If there's an error during raster data processing or polygon creation.
        """
    with rasterio.open(InferenceConfig.PATH_DTM) as dtm_src, rasterio.open(InferenceConfig.PATH_DSM) as dsm_src:
        dtm_data = dtm_src.read(1)
        dsm_data = dsm_src.read(1)
        transform = dtm_src.transform  # does not matter which, since they must have been aligned before
        crs_name = dtm_src.crs.data['init']
        pixel_size_x = transform.a
        pixel_size_y = abs(transform.e)

        height_mask = create_height_difference_mask(dsm_data, dtm_data, InferenceConfig.THRESHOLD_HEIGHT)

        # load convex hull and corresponding multipolygons
        original_multipolygons, convex_hull_polygons = load_polygons_in_correct_order(
            InferenceConfig.DIRECTORY_POLYGONS,
            InferenceConfig.DIRECTORY_POLYGONS_CONVEX_HULL)

        for i, (original_multipolygon, polygon_convex) in enumerate(zip(original_multipolygons, convex_hull_polygons)):
            # Create exterior mask of the original multipolygon to get the base height
            exterior_list = [polygon.exterior for polygon in original_multipolygon.geoms]
            combined_exterior = unary_union(exterior_list)
            original_exterior_mask = geometry_mask([combined_exterior], out_shape=dtm_data.shape,
                                                   transform=transform, invert=True)

            # Create mask for the original stockpile heap (dsm data)
            original_multipolygon_mask = geometry_mask(original_multipolygon.geoms, out_shape=dsm_data.shape,
                                                       transform=transform, invert=True)

            # Create intersection mask for the conveyor belt areas
            convex_hull_polygon_mask = geometry_mask([polygon_convex], out_shape=dtm_data.shape, transform=transform,
                                                     invert=True)

            intersection_convex_hull_height_mask = np.logical_and(convex_hull_polygon_mask, height_mask)

            # if the height mask and polygon mask have no intersection,
            # we don't have anything in the sand heap. if they do,
            # we have i.e. conveyor belts above them, where we need dtm values
            if intersection_convex_hull_height_mask.sum() != 0:
                create_volume_polygons_with_dsm_and_dtm_values(InferenceConfig.DIRECTORY_POLYGONS_FINAL,
                                                               original_multipolygon, dtm_data,
                                                               dsm_data, InferenceConfig.PADDING_DTM,
                                                               InferenceConfig.PADDING_DSM,
                                                               original_multipolygon_mask,
                                                               intersection_convex_hull_height_mask,
                                                               original_exterior_mask,
                                                               pixel_size_x, pixel_size_y,
                                                               i, crs_name, InferenceConfig.SAND_DENSITY)
            else:
                create_volume_polygons_with_dsm_values(InferenceConfig.DIRECTORY_POLYGONS_FINAL, original_multipolygon,
                                                       dtm_data, dsm_data, InferenceConfig.PADDING_DTM,
                                                       InferenceConfig.PADDING_DSM,
                                                       original_multipolygon_mask, original_exterior_mask,
                                                       pixel_size_x, pixel_size_y,
                                                       i, crs_name, InferenceConfig.SAND_DENSITY)


def save_multipolygons_for_leaflet():
    """
      Transforms and saves multipolygons as GeoJSON files for Leaflet map display.

      Loads multipolygons and associated GeoJSON from a directory, transforms them from EPSG:32632
      (UTM Zone 32N) to WGS84 coordinate system, updates centroid coordinates, and saves each
      transformed multipolygon as a GeoJSON file.
    """
    multipolygons, geojson_multipolygons = load_final_multipolygons_and_geojson(
        InferenceConfig.DIRECTORY_POLYGONS_FINAL)

    transformed_multipolygons = [transform_epsg32632_geometry_to_wgs84(multipolygon) for multipolygon in multipolygons]

    for i, transformed_multipolygon in enumerate(transformed_multipolygons):
        transformed_centroid = transform_epsg32632_geometry_to_wgs84(
            Point(geojson_multipolygons[i]["properties"]["centroid"]))

        geojson_multipolygons[i]["properties"]["centroid"] = [transformed_centroid.x, transformed_centroid.y]

        geojson_dict = mapping(transformed_multipolygon)

        geojson_feature = {
            "type": "Feature",
            "properties": geojson_multipolygons[i]["properties"],
            "geometry": geojson_dict

        }

        geojson = {
            "type": "FeatureCollection",
            "features": [geojson_feature]
        }

        os.makedirs(InferenceConfig.DIRECTORY_LEAFLET_POLYGONS, exist_ok=True)

        with open(os.path.join(InferenceConfig.DIRECTORY_LEAFLET_POLYGONS, f"{i}.geojson"), 'w') as f:
            json.dump(geojson, f)


if __name__ == '__main__':
    InferenceConfig.initialize(100 * 100, 1.5)
    # Step 0: Align DTM and DSM (if not already aligned)
    try_to_align_rasters()

    # Step 1: Create the Tensor Patches
    create_tensor_patches()
    #
    # # Step 2: Create the Model Predictions for the Tensor Patches
    create_model_predictions()

    # Step 3: Merge the Predictions
    create_merged_predictions()

    # Step 4: Optionally visualise the merged array and volume calculation
    # visualise_merging_results(InferenceConfig.PATH_MERGED_NP_ARRAY)

    # Step 5: Generate the resulting Qgis Polygons
    create_qgis_multipolygons_with_merged_predictions()

    # Step 6: Generate the Convex Hulls of these Polygons
    create_convex_hull_polygons()

    # Step 7: Optionally visualise height difference and polygons
    # visualise_diffed_dtm_dsm()
    # create_dtm_dsm_polygons_for_visualisation()

    # Step 8: Calculate Volume
    create_polygons_with_volume()

    # Step 9: Save Polygons in correct format for Leaflet
    save_multipolygons_for_leaflet()
