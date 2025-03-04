import cv2
import torch
from rasterio.enums import Resampling
from rasterio.warp import reproject
import numpy as np
import rasterio
from osgeo import gdal
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RANSACRegressor

from utils import normalise_image_tensor_data_0_to_1


def align_dem_to_orthophoto(path_dem: str, path_orthophoto: str, path_output: str):
    """
    Aligns a Digital Elevation Model (DEM) to an orthophoto and saves the aligned DEM.

    Args:
        path_dem (str): File path to the DEM raster dataset.
        path_orthophoto (str): File path to the orthophoto raster dataset.
        path_output (str): File path to save the aligned DEM.

    Raises:
        IOError: If there is an issue opening or reading the raster files,
                 or writing to the output raster file.
    """
    try:
        with rasterio.open(path_dem) as src, rasterio.open(path_orthophoto) as ref:
            ref_transform = ref.meta['transform']
            src_transform = src.meta['transform']
            with rasterio.open(path_output, 'w', driver='GTiff',
                               width=ref.width, height=ref.height,
                               count=src.count, dtype=src.dtypes[0],
                               crs=ref.crs, transform=ref_transform, nodata=src.meta['nodata']) as dst:
                for i in range(1, src.count + 1):
                    reproject(source=rasterio.band(src, i),
                              destination=rasterio.band(dst, i),
                              src_transform=src_transform,
                              src_crs=src.crs,
                              dst_transform=ref_transform,
                              dst_crs=ref.crs,
                              resampling=Resampling.nearest,
                              dst_nodata=src.meta['nodata'])
    except Exception as e:
        raise IOError(f"Error aligning DEM to orthophoto: {e}")


def check_alignment_tif_files(tif_src_path_1: str, tif_src_path_2: str) -> bool:
    """
        Checks if two raster datasets are aligned in terms of shape, CRS, and transformation.

        Args:
            tif_src_path_1 (str): File path to the first raster dataset.
            tif_src_path_2 (str): File path to the second raster dataset.

        Returns:
            bool: True if the datasets are aligned, False otherwise.

        Raises:
            IOError: If there is an issue opening or reading the raster files,
                 or writing to the output raster file.
    """
    try:
        with rasterio.open(tif_src_path_1) as src1, rasterio.open(tif_src_path_2) as src2:
            if src1.shape != src2.shape:
                return False

            if src1.crs != src2.crs:
                return False

            if src1.transform != src2.transform:
                return False

            return True
    except Exception as e:
        raise IOError(f"Error checking alignment of TIFF files: {e}")


def check_equality_tif_files(tif_src_path_1: str, tif_src_path_2: str, only_compare_common_bands: bool = False) -> bool:
    """
    Checks if two raster datasets are equal in terms of pixel values.

    Args:
        tif_src_path_1 (str): File path to the first raster dataset.
        tif_src_path_2 (str): File path to the second raster dataset.
        only_compare_common_bands (bool): Flag to indicate whether to compare only common bands (default: False).

    Returns:
        bool: True if the datasets are equal, False otherwise.

    Raises:
        IOError: If there is an issue opening or reading the raster files,
                 or writing to the output raster file.
    """
    try:
        if not check_alignment_tif_files(tif_src_path_1, tif_src_path_2):
            return False

        with rasterio.open(tif_src_path_1) as src1, rasterio.open(tif_src_path_2) as src2:
            if only_compare_common_bands:
                min_bands = min(src1.count, src2.count)
                bands_src1 = src1.read(range(1, min_bands + 1))
                bands_src2 = src2.read(range(1, min_bands + 1))
                return np.array_equal(bands_src1, bands_src2)
            bands_src1 = src1.read()
            bands_src2 = src2.read()
            return np.array_equal(bands_src1, bands_src2)
    except Exception as e:
        raise IOError(f"Error checking equality of TIFF files: {e}")


def create_slope_raster_from_dem(path_dem: str, path_output: str):
    """
    Creates a slope raster from a Digital Elevation Model (DEM) using GDAL.

    Args:
        path_dem (str): File path to the DEM raster dataset.
        path_output (str): File path to save the slope raster.

    Raises:
        IOError: If there is an issue opening or reading the raster files,
                 or writing to the output raster file.
    """
    try:
        dem_input = gdal.Open(path_dem)
        gdal.DEMProcessing(path_output, dem_input, "slope", computeEdges=True)
    except Exception as e:
        raise IOError(f"Error creating slope raster from DEM: {e}")


def create_blend_rgb_ortho_with_dem_raster(path_rgb_ortho: str, path_dem_raster: str, path_output: str,
                                           blend_alpha_ortho: float):
    """
    Creates a blended RGB orthophoto with a DEM raster.

    Args:
        path_rgb_ortho (str): File path to the RGB orthophoto raster dataset.
        path_dem_raster (str): File path to the DEM raster dataset.
        path_output (str): File path to save the blended orthophoto.
        blend_alpha_ortho (float): Alpha blending coefficient for orthophoto (0.0 to 1.0).

    Raises:
        IOError: If there is an issue opening or reading the raster files,
                 or writing to the output raster file.
    """
    try:
        with rasterio.open(path_rgb_ortho) as ortho_src, rasterio.open(path_dem_raster) as dsm_src:
            # read data
            ortho_np = ortho_src.read()[:3].transpose(1, 2, 0)  # [H, W, C]
            meta = ortho_src.meta
            # discard alpha channel as it is not necessary
            meta.update(count=3)

            # transpose to correct format for np
            dsm_np = dsm_src.read()  # [C, H , W]
            dsm_no_data = dsm_src.meta['nodata']
            dsm_tensor = torch.from_numpy(dsm_np)

            # normalise height tensors
            height_tensor_normalised = normalise_image_tensor_data_0_to_1(dsm_tensor, no_data_value=dsm_no_data)

            height_np_normalised = height_tensor_normalised.numpy(force=True).transpose((1, 2, 0))

            # clip to interval
            blended_image_uint8_hwc = create_blending_dem_ortho(height_np_normalised, ortho_np, blend_alpha_ortho)

            # Save the processed ortho image
            with rasterio.open(path_output, 'w', **meta) as dst:
                for i in range(3):
                    dst.write(blended_image_uint8_hwc[:, :, i], i + 1)
    except Exception as e:
        raise IOError(f"Error blending RGB orthophoto with DEM raster: {e}")


def create_blending_dem_ortho(dem_normalised_np: np.ndarray, ortho_np: np.ndarray, blending_alpha: float):
    """
        Create a blended image from a normalized DEM (Digital Elevation Model) and an orthophoto based
        on a defined blending_alpha.

        Args:
            dem_normalised_np (np.ndarray): Normalized DEM data as a NumPy array of shape [H, W, C].
            ortho_np (np.ndarray): Orthophoto data as a NumPy array of shape [H, W, C].
            blending_alpha (float): Alpha blending factor between [0, 1].

        Returns:
            np.ndarray: Blended image as a NumPy array of dtype np.uint8 and shape [H, W, C].
    """
    # stack dem data
    elev_data_broadcasted = np.broadcast_to(dem_normalised_np, ortho_np.shape)

    # cast normalized data to uint8
    elev_data = (elev_data_broadcasted * 255).astype(np.uint8)

    # blend
    blended_image = (blending_alpha * ortho_np) + ((1 - blending_alpha) * elev_data)

    # clip to interval
    blended_image_uint8 = np.clip(blended_image, 0, 255).astype(np.uint8)

    return blended_image_uint8


def get_no_data_value(path_raster: str) -> float:
    """
    Retrieves the nodata value of a raster dataset.

    Args:
        path_raster (str): File path to the raster dataset.

    Returns:
        float: Nodata value of the raster dataset.
    """
    try:
        with rasterio.open(path_raster) as src:
            return src.nodata
    except Exception as e:
        raise IOError(f"Error retrieving nodata value from raster: {e}")


if __name__ == '__main__':
    print("")
    # create_diffed_dtm("resources/Original_Ortho_DSM_DTM/aligned_dtm.tif", "resources/Original_Ortho_DSM_DTM/aligned_dsm.tif",
    #                   "test.tif", 2)
    # median_filter("train_dsm_slope.tif", "holla2")
    # print(estimate_ground_height2("resources/Split_Ortho_DSM_DTM/train_dsm.tif", "test_it_in.tif"))
    # create_binary_height_raster(path_dem="resources/Split_Ortho_DSM_DTM/train_dsm.tif", path_output="here.tif")
    create_blend_rgb_ortho_with_dem_raster(path_rgb_ortho="inference/inputs/orthophoto.tif",
                                           path_dem_raster="inference/inputs/dsm.tif",
                                           path_output="ortho_dsm_blended_tiff_a06.tif",
                                           blend_alpha_ortho=0.6)
    # create_slope_raster_from_dem("resources/Original_Ortho_DSM_DTM/aligned_dsm.tif", "aligned_dsm_slope.tif")
    # print(check_alignment_tif_files("resources/Original_Ortho_DSM_DTM/orthophoto.tif",
    #                                 "resources/Original_Ortho_DSM_DTM/aligned_dsm.tif"))
    # align_dem_to_orthophoto("resources/Original_Ortho_DSM_DTM/dtm.tif",
    #                         "resources/Original_Ortho_DSM_DTM/orthophoto.tif",
    #                         "resources/Original_Ortho_DSM_DTM/aligned_dtm.tif")
