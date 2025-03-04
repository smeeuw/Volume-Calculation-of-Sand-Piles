import os
from typing import Union, Tuple, Dict, List, Optional

import matplotlib
import numpy as np
import torch
from rasterio._base import Affine
from rasterio.features import geometry_mask
from rasterio.merge import merge
import rasterio
from rasterio.windows import Window, from_bounds
from shapely import MultiPolygon, Polygon

from utils import normalise_image_tensor_data_0_to_1


def split_tiff_image_into_tiff_patches(path_tiff_image: str, location_folder: str,
                                       fill_value_padding: Union[int, float],
                                       patch: Tuple[int, int] = (512, 512),
                                       kept_number_of_bands: int = None, overlap_in_px: int = None,
                                       remove_padded_pixels_threshold: float = None) -> None:
    """
    Splits a TIFF image into smaller TIFF patches.

    Parameters:
        path_tiff_image (str): The path to the input TIFF image.
        location_folder (str): The folder location to save the TIFF patches.
        fill_value_padding (Union[int, float]): The value to use for padding in the patches. Can deal with height data from DEM or RGB-Data from Orthophoto.
        patch (Tuple[int, int], optional): The size of the patches to create, in pixels. Defaults to (512, 512).
        kept_number_of_bands (int, optional): The number of bands to keep from the input image. Defaults to None, which keeps all bands.
        overlap_in_px (int, optional): The amount of overlap between patches, in pixels. Defaults to None, which means no overlap.
        remove_padded_pixels_threshold (float, optional): The threshold for removing patches with too many padded pixels. Defaults to None, which means no removal threshold. If a value lower than 1 is chosen, spatial information will be lost when the patches are merged. It is only meant for training, where the spatial information is not relevant.

    Raises:
        FileNotFoundError: If the specified path_tiff_image does not exist.
        AssertionError: If the overlap_in_px is greater than or equal to the patch size.

    Notes:
        - The default patch size is (512, 512) pixels.
        - If no overlap is specified, patches will not overlap.
        - If remove_padded_pixels_threshold is set, patches with too many padded pixels will be skipped.

    Returns:
        None
    """

    if overlap_in_px is not None:
        assert overlap_in_px < patch[0] and overlap_in_px < patch[
            1], "Overlap in pixels must be smaller than patch size"

    os.makedirs(location_folder, exist_ok=True)

    with rasterio.open(path_tiff_image, 'r') as src:

        if kept_number_of_bands is None:
            kept_number_of_bands = src.meta['count']

        image_width = src.meta['width']
        image_height = src.meta['height']
        id_counter = 0
        result_width, result_height, step_size_width, step_size_height = get_result_dimensions(image_width,
                                                                                               image_height, patch,
                                                                                               overlap_in_px)
        for y in range(0, result_height, step_size_height):
            for x in range(0, result_width, step_size_width):
                # do not take last step twice. happens in the overlap case because step size is smaller than patch size,
                # so remainder of last patch is taken twice
                if x + patch[0] > result_width or y + patch[1] > result_height:
                    continue

                window = Window(x, y, patch[0], patch[1])

                data = src.read(window=window, boundless=True, fill_value=fill_value_padding)

                if remove_padded_pixels_threshold is not None:
                    if number_of_padded_pixels_above_threshold(data=data, patch_size=patch,
                                                               padding_value=fill_value_padding,
                                                               remove_padded_pixels_threshold=remove_padded_pixels_threshold):
                        continue

                transform = src.window_transform(window)

                meta = src.meta.copy()
                meta.update({
                    "height": patch[1],
                    "width": patch[0],
                    "transform": transform,
                    "count": kept_number_of_bands
                })

                save_image_patch(data, meta, location_folder, id_counter, kept_number_of_bands)

                id_counter += 1


def merge_patches(location_folder: str, result_folder: str, result_filename: str,
                  original_meta: Dict[str, any]) -> None:
    """
        Merge patches from a location folder into a single raster image and save the result.

        Parameters:
            location_folder (str): The folder containing individual patch files.
            result_folder (str): The folder where the merged result will be saved.
            result_filename (str): The filename (without extension) of the merged result.
            original_meta (Dict[str, any]): Metadata of the original raster image.

        Raises:
            FileNotFoundError: If the specified location_folder does not exist.

        Notes:
            - This function assumes that the patch files are in GeoTIFF format.
            - This function also works if the patches overlap.
            - The original_meta dictionary is expected to contain metadata properties such as 'nodata', 'height',
              'width', 'transform', etc.
        """
    patches = []
    for patch_path in os.listdir(location_folder):
        src = rasterio.open(os.path.join(location_folder, patch_path))
        patches.append(src)

    no_data_value = original_meta['nodata']

    mosaic, out_trans = merge(patches, nodata=no_data_value)
    original_bounds = rasterio.transform.array_bounds(
        original_meta['height'], original_meta['width'], original_meta['transform']
    )
    window = from_bounds(*original_bounds, transform=out_trans)
    window = window.round_offsets()
    window = window.round_lengths()

    mosaic_cropped = mosaic[:, window.row_off: window.row_off + window.height,
                     window.col_off: window.col_off + window.width]

    meta = original_meta.copy()
    meta.update({
        "height": mosaic_cropped.shape[1],
        "width": mosaic_cropped.shape[2],
        "transform": out_trans,
        "count": mosaic_cropped.shape[0]
    })

    os.makedirs(result_folder, exist_ok=True)

    with rasterio.open(f"{os.path.join(result_folder, result_filename)}.tif", 'w', **meta) as dst:
        dst.write(mosaic_cropped)

    for src in patches:
        src.close()


def get_result_dimensions(image_width: int, image_height: int, patch: Tuple[int, int], overlap_in_px: int = None) -> \
        Tuple[int, int, int, int]:
    """
    Calculate the dimensions and step sizes for splitting an image into patches.

    Parameters:
        image_width (int): The width of the input image, in pixels.
        image_height (int): The height of the input image, in pixels.
        patch (Tuple[int, int]): The size of the patches to create, specified as (width, height) in pixels.
        overlap_in_px (int, optional): The amount of overlap between patches, in pixels. Defaults to None, which means no overlap.

    Returns:
        Tuple[int, int, int, int]: A tuple containing the following values:
            - result_width (int): The width of the result image, considering patch size and overlap.
            - result_height (int): The height of the result image, considering patch size and overlap.
            - step_size_width (int): The step size for moving horizontally when creating patches.
            - step_size_height (int): The step size for moving vertically when creating patches.
    """
    if overlap_in_px is not None:
        step_size_height = patch[1] - overlap_in_px
        step_size_width = patch[0] - overlap_in_px
        result_width, result_height = find_image_size_with_overlap(image_width=image_width,
                                                                   image_height=image_height,
                                                                   step_size_width=step_size_width,
                                                                   step_size_height=step_size_height,
                                                                   patch_size_height=patch[1],
                                                                   patch_size_width=patch[0])
    else:
        step_size_height = patch[1]
        step_size_width = patch[0]
        result_width = find_nearest_multiple(number=step_size_width, target=image_width)
        result_height = find_nearest_multiple(number=step_size_height, target=image_height)
    return result_width, result_height, step_size_width, step_size_height


def create_masked_rgb_orthophoto(path_orthophoto: str, path_output_folder: str, filename: str,
                                 multipolygons: List[MultiPolygon],
                                 colors: Optional[List[Tuple[int, int, int]]] = None,
                                 pad_value: Optional[int] = None) -> None:
    """
        Create a masked RGB orthophoto by overlaying polygons on an input orthophoto. The input can also be patches of an orthophoto.
        In that case, a pad_value should be specified, since it is necessary to prevent padded parts from being masked with a polygon.

        Parameters:
            path_orthophoto (str): The path to the input orthophoto file.
            path_output_folder (str): The path to the folder where the output orthophoto will be saved.
            filename (str): The name of the output orthophoto file (without extension).
            multipolygons (list): A list of Shapely MultiPolygon objects representing the polygons to overlay on the orthophoto.
            colors (list, optional): A list of RGB tuples representing the colors for each multipolygon.
                                     If not provided, colors will be generated automatically.
            pad_value (int, optional): The value that was used for padding for input in the form of patches.

        Returns:
            None

        Notes:
            - The output orthophoto will be saved as a GeoTIFF file in the specified output folder.
            - If colors are not provided, a color palette will be automatically generated based on the number of multipolygons.
            - The polygons are overlaid on the orthophoto by changing the intensity of pixels within the polygon boundaries.
            - If pad_value is specified, padded values in the patches will not be filled with polygon masks.
        """
    os.makedirs(path_output_folder, exist_ok=True)
    with rasterio.open(path_orthophoto, 'r') as ortho_src:
        red_band = ortho_src.read(1)
        green_band = ortho_src.read(2)
        blue_band = ortho_src.read(3)
        polygons = []
        if colors is None:
            colors = generate_color_palette(len(multipolygons))
            colors = convert_colors_to_0_to_255(colors)
        for i, multipolygon in enumerate(multipolygons):
            for polygon in multipolygon.geoms:
                change_intensity_of_pixels_in_polygon(red_band, green_band, blue_band, transform=ortho_src.transform,
                                                      polygon=polygon, new_rgb_color=colors[i],
                                                      pad_value=pad_value)
                polygons.append(polygon)
        meta = ortho_src.meta.copy()
        meta.update({
            "count": 3
        })
        combined_data = np.stack([red_band, green_band, blue_band], axis=-1)
        with rasterio.open(f"{os.path.join(path_output_folder, filename)}.tif", 'w', **meta) as dst:
            dst.write(combined_data[:, :, :3].transpose(2, 0, 1))


# mainly for documentation and sanity check
def create_masked_tif_patches(path_folder_patches: str, path_output_folder: str, output_filename: str,
                              multipolygons: List[MultiPolygon], pad_value: int) -> None:
    """
    Create masked patches from a folder of TIFF images. A pad_value should be specified,
    since it is necessary to prevent padded parts from being masked with a polygon. Polygons are based
    on real world coordinates in a reference system and, since padded pixels are as well, they can extend
    to these padded pixels.

    Parameters:
        path_folder_patches (str): Path to the folder containing TIFF image patches.
        path_output_folder (str): Path to the folder where the masked patches will be saved.
        output_filename (str): Base name for the output masked patches.
        multipolygons (List[MultiPolygon]): List of Shapely MultiPolygon objects representing the areas to mask.
        pad_value (int): The value that was used for padding for input in the form of patches.

    Returns:
        None

    Notes:
        - The function reads TIFF image patches from the specified folder.
        - Each TIFF image patch is processed to create a masked RGB orthophoto using the provided multipolygons.
        - The output masked patches are saved in the specified output folder with filenames based on the output_filename
          and an incremental counter.
    """
    os.makedirs(path_output_folder, exist_ok=True)
    tif_filename_patches = [f for f in os.listdir(path_folder_patches) if f.endswith('.tif')]
    id_counter = 0
    colors = generate_color_palette(len(multipolygons))
    colors = convert_colors_to_0_to_255(colors)
    for file_name_patch in tif_filename_patches:
        create_masked_rgb_orthophoto(os.path.join(path_folder_patches, file_name_patch), path_output_folder,
                                     f"{output_filename}_{id_counter}",
                                     multipolygons, colors, pad_value)
        id_counter += 1


# padded pixels should never contain polygons!
# This can happen because masks are based on real world coordinates and extend to padded pixels
def change_intensity_of_pixels_in_polygon(r_band: np.ndarray, g_band: np.ndarray, b_band: np.ndarray,
                                          transform: Affine, polygon: Polygon, pad_value: int = None,
                                          new_rgb_color: Tuple[int, int, int] = (255, 0, 0)) -> None:
    """
    Change the intensity of pixels within a polygon in RGB bands.

    Parameters:
        r_band (np.ndarray): Array representing the red band of the image.
        g_band (np.ndarray): Array representing the green band of the image.
        b_band (np.ndarray): Array representing the blue band of the image.
        transform (Affine): Affine transformation from real-world coordinates to pixel coordinates.
        polygon (Polygon): Shapely Polygon representing the area where pixel intensities will be changed.
        pad_value (int, optional): Value indicating padded pixels. Defaults to None.
        new_rgb_color (Tuple[int, int, int], optional): RGB color tuple to fill the polygon. Defaults to (255, 0, 0).

    Returns:
        None

    Notes:
        - This function modifies the input arrays in-place.
        - The function modifies pixel intensities within the specified polygon to the new RGB color.
        - If pad_value is provided, pixels with this value are considered as padded and excluded from the polygon area.
        - The pixel intensities within the polygon are changed only in RGB bands.
        - The transform parameter is used to convert real-world coordinates of the polygon to pixel coordinates.
    """
    assert r_band.shape == g_band.shape == b_band.shape
    polygon_mask = geometry_mask([polygon], out_shape=r_band.shape, transform=transform, invert=True)
    padded_mask = np.ones(r_band.shape, dtype=bool)
    if pad_value is not None:
        # remove masks from padded pixels
        padded_mask = ~ ((r_band == pad_value) & (g_band == pad_value) & (b_band == pad_value))
    combined_mask = polygon_mask & padded_mask
    r_band[combined_mask] = new_rgb_color[0]
    g_band[combined_mask] = new_rgb_color[1]
    b_band[combined_mask] = new_rgb_color[2]


def generate_color_palette(num_colors, colormap_name='tab20') -> List[Tuple[float, float, float]]:
    """
        Generate a color palette with the specified number of colors using a colormap.

        Parameters:
            num_colors (int): The number of colors to generate for the palette.
            colormap_name (str, optional): The name of the colormap to use. Defaults to 'tab20'.

        Returns:
            List[Tuple[float, float, float]]: A list of RGB tuples representing the colors in the palette.

        Notes:
            - This function uses Matplotlib's colormaps to generate colors.
            - The colormap is sampled to obtain the specified number of colors.
            - The resulting color palette is represented as a list of RGB tuples, where each tuple contains
              three float values between 0 and 1 representing the red, green, and blue components of the color.
        """
    colormap = matplotlib.colormaps[colormap_name]
    colors = [colormap(i)[:-1] for i in range(num_colors)]
    return colors


def convert_colors_to_0_to_255(list_colors: List[Tuple[float, float, float]]) -> List[Tuple[int, int, int]]:
    """
        Convert colors from the range [0, 1] to the range [0, 255].

        Parameters:
            list_colors (List[Tuple[float, float, float]]): A list of RGB tuples where each component ranges from 0 to 1.

        Returns:
            List[Tuple[int, int, int]]: A list of RGB tuples where each component ranges from 0 to 255.

        Notes:
            - This function converts RGB color values from the range [0, 1] to the range [0, 255].
            - Each tuple in the input list represents an RGB color, with three float values between 0 and 1
              representing the red, green, and blue components.
            - The output list contains the same RGB colors, but with each component converted to an integer in the
              range [0, 255].
        """
    new_colors = []
    for color in list_colors:
        new_colors.append((int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
    return new_colors


def number_of_padded_pixels_above_threshold(data: np.ndarray, patch_size: Tuple[int, int],
                                            padding_value: Union[int, float],
                                            remove_padded_pixels_threshold: float) -> bool:
    """
    Check if the number of padded pixels in the data exceeds a specified threshold.

    Parameters:
        data (np.ndarray): The input data array [C, H, W].
        patch_size (Tuple[int, int]): The size of the patch in pixels.
        padding_value (Union[int, float]): The value indicating padded pixels.
        remove_padded_pixels_threshold (float): The threshold percentage of padded pixels to consider for removal.

    Returns:
        bool: True if the number of padded pixels exceeds the threshold, False otherwise.

    Notes:
        - The threshold should be given as a percentage, with values between 0 and 1.
    """
    if isinstance(padding_value, float):
        padded_pixels = np.sum(np.all(np.isclose(data, padding_value), axis=0))
    elif isinstance(padding_value, int):
        padded_pixels = np.sum(np.all(data == padding_value, axis=0))
    else:
        raise Exception('Invalid type for padding_value. Must be either float or int.')
    number_pixels = patch_size[0] * patch_size[1]
    if padded_pixels / number_pixels >= remove_padded_pixels_threshold:
        return True
    return False


def save_image_patch(data: np.ndarray, meta: Dict[str, any], location_folder: str,
                     id_counter: int, kept_number_of_bands: int) -> None:
    """
    Save an image patch to a GeoTIFF file based on a counter that is incremented by one.

    Parameters:
        data (np.ndarray): The image patch data array.
        meta (Dict[str, any]): Metadata dictionary for the GeoTIFF file.
        location_folder (str): The folder location where the GeoTIFF file will be saved.
        id_counter (int): An identifier for the image patch.
        kept_number_of_bands (int): The number of bands to keep when saving the image patch.

    Returns:
        None

    Notes:
        - The original_meta dictionary is expected to contain metadata properties such as 'nodata', 'height', width', 'transform', etc.
    """
    with rasterio.open(os.path.join(location_folder, f"{id_counter}.tif"), 'w',
                       **meta) as dst:
        dst.write(data[:kept_number_of_bands, :, :])


def convert_temporary_tiff_patches_to_torch_tensors(path_folder_tiff_patches: str,
                                                    path_result_folder: str,
                                                    kept_number_of_bands: int = None,
                                                    normalize: bool = False,
                                                    no_data_value: Union[int, float] = None) -> torch.Tensor:
    """
    Converts TIFF patches in a specified folder to a single Torch tensor and saves it to disk.
    Also normalizes the resulting Tensor to [0, 1] if `normalize` is `True`.
    After saving, deletes all TIFF files in the folder.

    Args:
        path_folder_tiff_patches (str): The path to the folder containing the TIFF files.
        path_result_folder (str): The path of the folder where the resulting Torch Tensors will be saved.
        kept_number_of_bands (int, optional): The number of bands to keep from each TIFF file.
         If None, all bands from each TIFF file will be kept. Defaults to None.
        normalize(bool): Specifies whether to normalize the resulting Tensor to [0, 1]. Defaults to False.
        no_data_value (Union[int, float]): No data values to mask during normalization.

    Raises:
        AssertionError: If no TIFF files are found in the specified folder.
    """
    tiff_files = [os.path.join(path_folder_tiff_patches, f) for f in os.listdir(path_folder_tiff_patches) if
                  f.endswith('.tif')]

    tiff_files = sorted(tiff_files, key=lambda x: int(os.path.basename(os.path.splitext(x)[0])))

    assert len(tiff_files) > 0, 'No tiff files found in {}'.format(path_folder_tiff_patches)

    image_list = []

    for tiff_file in tiff_files:
        with rasterio.open(tiff_file) as src:
            if kept_number_of_bands is None:
                kept_number_of_bands = src.meta['count']
            img_data = src.read(indexes=list(range(1, kept_number_of_bands + 1)))
            image_list.append(torch.tensor(img_data))

    os.makedirs(path_result_folder, exist_ok=True)
    idx = 0

    for i, image_tensor in enumerate(image_list):
        if normalize:
            image_list[i] = normalise_image_tensor_data_0_to_1(image_list[i], no_data_value=no_data_value)
        torch.save(image_list[i], os.path.join(path_result_folder, f"{idx}.pt"))
        idx += 1

    # Delete temporary files
    for tiff_file in tiff_files:
        os.remove(tiff_file)

    return torch.stack(image_list)


def find_nearest_multiple(number: int, target: int) -> int:
    """
    Find the nearest multiple of a number that is greater than or equal to a target.

    Parameters:
        number (int): The number to find multiples of.
        target (int): The target value.

    Returns:
        int: The nearest multiple of the input number that is greater than or equal to the target value.

    Raises:
        AssertionError: If the input number is not positive or exceeds the target value.
    """

    assert 0 < number <= target, "Only works for positive inputs and number <= target"
    if target % number == 0:
        return target
    else:
        return ((target // number) + 1) * number


def find_image_size_with_overlap(image_width: int, image_height: int, step_size_width: int,
                                 step_size_height: int, patch_size_width: int, patch_size_height: int) -> Tuple[
    int, int]:
    """
    Find the size of the resulting image considering that patches can overlap.

    Parameters:
        image_width (int): The width of the input image, in pixels.
        image_height (int): The height of the input image, in pixels.
        step_size_width (int): The step size for moving horizontally when creating patches.
        step_size_height (int): The step size for moving vertically when creating patches.
        patch_size_width (int): The width of the patches, in pixels.
        patch_size_height (int): The height of the patches, in pixels.

    Returns:
        Tuple[int, int]: The width and height of the resulting image considering that patches can overlap.
    """
    width = patch_size_width
    while width < image_width:
        width += step_size_width

    height = patch_size_height
    while height < image_height:
        height += step_size_height

    return width, height


if __name__ == "__main__":
    print("hey")
    convert_temporary_tiff_patches_to_torch_tensors(path_folder_tiff_patches="resources/val_no_over",
                                                    path_result_folder="testie", kept_number_of_bands=3)
    # test = np.load("test.npy")
    # print(test.shape)
    # split_tiff_image_into_tiff_patches(path_tiff_image="resources/Split_Ortho_DSM_DTM/val_dsm.tif",
    #                                    location_folder="lacucarhja",
    #                                    location_start_filename="lacucarhja", fill_value_padding=-9999.0,
    #                                    patch=(512, 512),
    #                                    kept_number_of_bands=None, overlap_in_px=None, remove_padded_pixels_threshold=1)
    # meta = load_metadata_from_orthophoto("resources/Split_Ortho_DSM_DTM/val_ortho.tif")
    # merge_patches(location_folder="resources/val_no_over_masks", result_folder="resources/val_no_over_mask_merge_res",
    #               result_filename="merged_masks", original_meta=meta,
    #               no_data_value=0)
    # create_masked_tif_patches(path_folder_patches="resources/val_no_over",
    #                           path_output_folder="resources/val_no_over_masks", output_filename="val_no_over_mask",
    #                           multipolygons=load_list_of_qgis_multipolygons_from_folder("resources/Polygons_Stockpile"), pad_value=0)
