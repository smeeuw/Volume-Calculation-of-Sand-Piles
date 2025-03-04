import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pyproj
import rasterio
import torch
from affine import Affine
from rasterio.features import geometry_mask
from shapely import Polygon, MultiPolygon, Point
from shapely.geometry import shape, mapping
from shapely.ops import transform

from mask_rcnn_eval import compute_binary_mask_iou


def load_tensors_in_correct_order_on_cpu(path_directory: str, stack: bool) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
       Load tensors from a directory in the correct numerical order. If the tensors are cuda tensors,
       they will be transformed to cpu tensors.

       Args:
           path_directory (str): The directory containing the tensor files.
           stack (bool): If True, stack the tensors into a single tensor. If False, return a list of tensors.

       Returns:
           Union[torch.Tensor, List[torch.Tensor]]: The loaded tensors, either as a single stacked tensor or a list of
           tensors.

       Raises:
           OSError: If there's an error reading tensor files from the directory.
           RuntimeError: If there's an error loading tensors.
       """
    tensor_files = [os.path.join(path_directory, f) for f in
                    os.listdir(path_directory) if
                    f.endswith('.pt')]

    tensor_files = sorted(tensor_files, key=lambda x: int(os.path.basename(os.path.splitext(x)[0])))

    tensors = []

    for tensor_file in tensor_files:
        tensor_path = os.path.join(tensor_file)
        tensor = torch.load(tensor_path)
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensors.append(tensor)

    if stack:
        return torch.stack(tensors)
    else:
        return tensors


def load_polygons_in_correct_order(path_polygons: str, path_convex_hull_polygons: str) -> Tuple[List[MultiPolygon],
List[Polygon]]:
    """
       Load original multipolygons and their corresponding convex hull polygons from directories.

       Args:
           path_polygons (str): The directory containing original multipolygon GeoJSON files.
           path_convex_hull_polygons (str): The directory containing convex hull polygon GeoJSON files.

       Returns:
           Tuple[List[Multipolygon], List[Polygon]]: A tuple containing lists of original multipolygons
           and convex hull polygons.

       Raises:
           OSError: If there's an error reading polygon files from the directories.
           ValueError: If there's an error parsing GeoJSON data or creating polygon shapes.
       """

    polygon_files = [os.path.join(path_polygons, f) for f in
                     os.listdir(path_polygons) if
                     f.endswith('.geojson')]

    polygon_convex_hull_files = [os.path.join(path_convex_hull_polygons, f) for f in
                                 os.listdir(path_convex_hull_polygons) if
                                 f.endswith('.geojson')]

    # sort both arrays so that files match

    polygon_files = sorted(polygon_files,
                           key=lambda x: int(os.path.basename(os.path.splitext(x)[0]).split('Multipolygon')[1]))
    polygon_convex_hull_files = sorted(polygon_convex_hull_files, key=lambda x: int(
        os.path.basename(os.path.splitext(x)[0]).split('Multipolygon')[1]))

    original_multipolygons = []
    convex_hull_polygons = []
    for original_multipolygon, convex_hull_polygon in zip(polygon_files, polygon_convex_hull_files):
        # load original multipolygon
        with open(original_multipolygon, 'r') as poly_source:
            geojson_data = json.load(poly_source)
            original_multipolygons.append(shape(geojson_data['features'][0]['geometry']))

        # load convex hulls
        with open(convex_hull_polygon, 'r') as poly_source:
            geojson_data = json.load(poly_source)
            convex_hull_polygons.append(shape(geojson_data['features'][0]['geometry']))

    return original_multipolygons, convex_hull_polygons


def save_polygon_to_disk(directory: str, filename: str, polygon: Union[Polygon, MultiPolygon], crs_name: str,
                         properties: Optional[Dict[str, Any]] = None):
    """
        Save a polygon to disk as a GeoJSON file.

        Args:
            directory (str): The directory where the GeoJSON file will be saved.
            filename (str): The name of the GeoJSON file.
            polygon (Union[Polygon, MultiPolygon]): The polygon to be saved.
            crs_name (str): The name of the coordinate reference system.
            properties(Optional[Dict[str, Any]]): An optional dictionary to specify properties, such as volume.

        Raises:
            OSError: If there's an error creating directories or writing to the file.
            ValueError: If there's an error serializing the polygon to GeoJSON format.
        """
    polygon_dict = {
        "type": "FeatureCollection",
        "name": filename.rsplit(".", 1)[0],
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "features": [{"type": "Feature", "geometry": mapping(polygon)}]
    }

    if properties is not None:
        polygon_dict.update(properties)

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename),
              'w') as f:
        json.dump(polygon_dict, f)


def check_correct_number_of_files_created(directory_ortho_tensors: str, directory_dsm_tensors: str,
                                          directory_dtm_tensors: str):
    """
        Check if the number of orthophoto tensors, DSM tensors, and DTM tensors is the same and bigger than zero.

        Args:
            directory_ortho_tensors (str): The directory containing orthophoto tensor files.
            directory_dsm_tensors (str): The directory containing DSM tensor files.
            directory_dtm_tensors (str): The directory containing DTM tensor files.

        Raises:
            AssertionError: If the number of files in the directories is unequal or zero.
        """
    fns_orthophoto = [os.path.join(directory_ortho_tensors, f) for
                      f in os.listdir(directory_ortho_tensors) if
                      f.endswith('.pt')]
    fns_dsm = [os.path.join(directory_dsm_tensors, f) for
               f in os.listdir(directory_dsm_tensors) if
               f.endswith('.pt')]
    fns_dtm = [os.path.join(directory_dtm_tensors, f) for
               f in os.listdir(directory_dtm_tensors) if
               f.endswith('.pt')]

    assert len(fns_dsm) == len(fns_orthophoto) == len(fns_dtm) > 0, ("Unequal number of patches generated"
                                                                     "make sure DTM, DSM, and Orthophoto are aligned")


def check_same_pixel_size_dsm_dtm_orthophoto(path_dsm: str, path_dtm: str, path_orthophoto: str,
                                             tolerance: float) -> bool:
    """
     Check if two raster datasets (DSM and DTM) have the same pixel size.

     Args:
         path_dsm (str): File path to the DSM raster dataset.
         path_dtm (str): File path to the DTM raster dataset.
         path_orthophoto (str): File path to the Orthophoto raster dataset.
         tolerance (float): Tolerance when checking for equality of the pixel size.

     Returns:
         bool: True if DSM and DTM have the same pixel size, False otherwise.
     """
    with rasterio.open(path_dtm) as dtm_src, rasterio.open(path_dsm) as dsm_src, rasterio.open(
            path_orthophoto) as ortho_src:
        transform_dtm = dtm_src.transform
        transform_dsm = dsm_src.transform
        transform_orthophoto = ortho_src.transform
        pixel_size_x_dtm = transform_dtm.a
        pixel_size_y_dtm = abs(transform_dtm.e)
        pixel_size_x_dsm = transform_dsm.a
        pixel_size_y_dsm = abs(transform_dsm.e)
        pixel_size_x_ortho = transform_orthophoto.a
        pixel_size_y_ortho = abs(transform_orthophoto.e)

        return (math.isclose(pixel_size_x_dtm, pixel_size_x_dsm, abs_tol=abs(tolerance)) and
                math.isclose(pixel_size_y_dtm, pixel_size_y_dsm, abs_tol=abs(tolerance)) and
                math.isclose(pixel_size_y_dtm, pixel_size_y_ortho, abs_tol=abs(tolerance)) and
                math.isclose(pixel_size_x_dtm, pixel_size_x_ortho, abs_tol=abs(tolerance)))


def create_dtm_polygons(output_directory: str, convex_hull_polygons: List[Polygon],
                        dtm_data: np.ndarray[float], transform: Affine, crs_name: str, height_mask: np.ndarray[bool]) -> \
        List[Tuple[int, Polygon]]:
    """
      Create DTM polygons from height mask within convex hull polygons. This method finds the intersection
      of the convex hull of a polygon and the height mask (which incorporates the conveyor belts).
      The result of this intersection is returned as Polygon.

      Args:
          output_directory (str): The directory to save the DTM polygon GeoJSON files.
          convex_hull_polygons (List[Polygon]): List of convex hull polygons.
          dtm_data (np.ndarray[float]): DTM data array.
          transform (Affine): Affine transformation of the DTM data.
          crs_name (str): Name of the coordinate reference system.
          height_mask (np.ndarray[bool]): Height mask that contains objects above the ground (i.e. conveyor belts).

      Returns:
          List[Tuple[int, Polygon]]: List of tuples containing index and DTM polygons.

      Raises:
          ValueError: If there's an error in creating the polygons.
      """
    resulting_polygons_dtm = []
    for i, polygon in enumerate(convex_hull_polygons):
        # generate mask of the convex hull
        convex_hull_polygon_mask = geometry_mask([polygon], out_shape=dtm_data.shape, transform=transform, invert=True)

        # generate intersection of height mask and convex hull mask
        # this is the area of the height mask inside the convex hull mask
        # it is the resulting polygon for the dtm values
        intersection_convex_hull_height_mask = np.logical_and(convex_hull_polygon_mask, height_mask)
        intersection_convex_hull_height_mask_int = np.uint8(intersection_convex_hull_height_mask) * 255

        # they must have an intersection, otherwise there is no conveyor belt in the sand heap
        if intersection_convex_hull_height_mask.sum() != 0:
            contours, _ = cv2.findContours(intersection_convex_hull_height_mask_int, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)  # only get largest contour

            resulting_polygon = create_crs_polygon_from_contour(largest_contour, transform)

            resulting_polygons_dtm.append((i, resulting_polygon))

            save_polygon_to_disk(output_directory, f"DTM_Multipolygon{i}.geojson", resulting_polygon,
                                 crs_name)

    return resulting_polygons_dtm


def create_dsm_polygons_with_dtm_and_original_multipolygons(output_directory: str,
                                                            dtm_polygons: List[Tuple[int, Polygon]],
                                                            original_multipolygons: List[MultiPolygon],
                                                            dtm_data: np.ndarray[float], transform: Affine,
                                                            crs_name: str, min_area_polygon: int):
    """
        Create DSM polygons with DTM polygons and original multipolygons. The original polygons represent
        the predictions of the model. The DTM polygons are the intersection of the convex hull of the
        original polygons and the objects above the instances (i.e. conveyor belts) which are highlighted in the height mask.
        The resulting DSM Polygons are the intersection of the DTM polygons and the original mask
        (so that no overlapping area is created). However, since the visualisation of the polygon depends on the
        exterior, there are no holes in the Polygons, which means the DTM polygons are not displayed as hole,
        even if they would theoretically create one. Despite that, the holes are considered when
        actually calculating the volume, so there never is any overlap in the volume calculation!

        Args:
            output_directory (str): The directory to save the DSM polygon GeoJSON files.
            dtm_polygons (List[Tuple[int, Polygon]]): List of tuples containing index and DTM polygons.
            original_multipolygons (List[MultiPolygon]): List of original multipolygons.
            dtm_data (np.ndarray[float]): DTM data array.
            transform (Affine): Affine transformation of the DSM data.
            crs_name (str): Name of the coordinate reference system.
            min_area_polygon (int): Minimum area threshold for polygons.

        Returns:
            None
        """
    for (i, polygon) in dtm_polygons:
        # convex mask of step before
        resulting_polygon_mask_dtm = geometry_mask([polygon], out_shape=dtm_data.shape,
                                                   transform=transform, invert=True)
        # generate mask for original polygons
        multipolygon_mask = geometry_mask(original_multipolygons[i].geoms, out_shape=dtm_data.shape,
                                          transform=transform, invert=True)

        # intersection of multipolygon and dtm polygon
        intersection_original_dtm = np.logical_and(resulting_polygon_mask_dtm, multipolygon_mask)

        # if nothing intersects, we do nothing,
        # since the case of untouched original multipolygons
        # should not be visualised
        if intersection_original_dtm.sum() != 0:
            multipolygon_mask[intersection_original_dtm] = 0
            dsm_polygon_mask = np.uint8(multipolygon_mask * 255)

            contours, _ = cv2.findContours(dsm_polygon_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            crs_polygons = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area_polygon:
                    crs_polygon = create_crs_polygon_from_contour(contour, transform)
                    crs_polygons.append(crs_polygon)

            # dtm values take all the space, so
            # no dsm polygon should be created
            if len(crs_polygons) == 0:
                continue

            resulting_multipolygon = MultiPolygon(crs_polygons)

            save_polygon_to_disk(output_directory, f"DSM_Multipolygon{i}.geojson", resulting_multipolygon,
                                 crs_name)


def create_crs_polygon_from_contour(contour: np.ndarray, transform: Affine) -> Polygon:
    """
       Convert a contour represented as a NumPy array to a Shapely Polygon object with coordinates transformed to CRS.
       It uses polygon.buffer(0) to tidy the polygon. If this is not enough to create a valid polygon, and Exception
       is thrown.

       Parameters:
           contour (np.ndarray): The contour as a NumPy array representing a list of points.
           transform (Affine): The affine transformation representing the coordinate transformation.

       Returns:
           Polygon: A Shapely Polygon object representing the contour with coordinates transformed to CRS.

       Raises:
           Exception: If the polygon is not valid after applying buffer(0) to tidy up the geometry.
       """
    contour = np.squeeze(contour)
    polygon = Polygon(contour)
    polygon = polygon.buffer(0)
    if isinstance(polygon, MultiPolygon):
        # very small parts that are not connected with rest of polygon (will be removed)
        polygon = max(polygon.geoms, key=lambda p: p.area)
    if not polygon.is_valid:
        raise Exception("Buffer zero trick did not work to tidy polygon. Geometries have to be fixed differently")

    polygon_crs_coords = [rasterio.transform.xy(transform, xy[1], xy[0]) for xy in polygon.exterior.coords]

    return Polygon(polygon_crs_coords)


def create_volume_polygons_with_dsm_and_dtm_values(directory_output: str, original_multipolygon: MultiPolygon,
                                                   dtm_data: np.ndarray[float], dsm_data: np.ndarray[float],
                                                   no_data_dtm: float, no_data_dsm: float,
                                                   original_multipolygon_mask: np.ndarray[bool],
                                                   intersection_convex_hull_height_mask: np.ndarray[bool],
                                                   original_exterior_mask: np.ndarray[bool],
                                                   pixel_size_x: float, pixel_size_y: float, i: int,
                                                   crs_name: str, density: float):
    """
    Create volume polygons with DSM and DTM values for a given original multipolygon. The original multipolygons
    represent the predictions of the model. The DTM polygons are the intersection of the convex hull of the
    original polygons and the objects above the instances (i.e. conveyor belts) which are
    highlighted in the height mask. The resulting DSM Polygons are the intersection of the DTM polygons and the original
    mask (so that no overlapping area is created). The base height of the resulting polygon is determined
    by the DTM height of the exterior of the multipolygon. The volume is calculated for values above and below
    the base height.

    Parameters:
        directory_output (str): The directory path to save the output files.
        original_multipolygon (MultiPolygon): The original multipolygon representing the predicted sand heap.
        dtm_data (np.ndarray[float]): The Digital Terrain Model (DTM) data.
        dsm_data (np.ndarray[float]): The Digital Surface Model (DSM) data.
        no_data_dtm (float): The no-data value for the DTM data.
        no_data_dsm (float): The no-data value for the DSM data.
        original_multipolygon_mask (np.ndarray[bool]): The mask representing the original multipolygon area.
        intersection_convex_hull_height_mask (np.ndarray[bool]): The mask representing the intersection between the
            convex hull of the original multipolygon and height mask.
        original_exterior_mask (np.ndarray[bool]): The mask representing the exterior polygon of the original
            multipolygon (model prediction). Used to calculate the base height of the resulting polygon.
        pixel_size_x (float): The pixel size in the x-direction.
        pixel_size_y (float): The pixel size in the y-direction.
        i (int): The index of the original multipolygon.
        crs_name (str): The name of the Coordinate Reference System (CRS).
        density (float): The density of the material.

    Returns:
        None
    """
    result_img_mask = np.zeros_like(dtm_data, dtype=np.uint8)
    # 1 is for dsm data, and 2 for dtm data. dtm data must overwrite dsm data
    # since we don't want to calculate the intersection twice
    result_img_mask[original_multipolygon_mask] = 1
    result_img_mask[intersection_convex_hull_height_mask] = 2

    dsm_data_result = dsm_data[result_img_mask == 1]
    dtm_data_result = dtm_data[result_img_mask == 2]

    min_dtm = calculate_base_height_with_exterior_polygon(dtm_data, no_data_dtm, original_exterior_mask)

    volume_dsm_above, volume_dsm_below = calculate_volume_above_and_below(dsm_data_result, no_data_dsm, min_dtm,
                                                                          pixel_size_x, pixel_size_y)
    volume_dtm_above, volume_dtm_below = calculate_volume_above_and_below(dtm_data_result, no_data_dtm, min_dtm,
                                                                          pixel_size_x, pixel_size_y)

    total_volume_m3_above = volume_dsm_above + volume_dtm_above
    total_volume_m3_below = volume_dsm_below + volume_dtm_below

    total_volume_t_above = total_volume_m3_above * density
    total_volume_t_below = total_volume_m3_below * density

    filename = (f"Multipolygon{i}_a{round(total_volume_m3_above, 2)}m3_a{round(total_volume_t_above, 2)}t_"
                f"b{round(total_volume_m3_below, 2)}m3_b{round(total_volume_t_below, 2)}t.geojson")

    properties_dict = create_properties_dict_volume_polygons(total_volume_m3_above, total_volume_t_above,
                                                             total_volume_m3_below, total_volume_t_below,
                                                             [original_multipolygon.centroid.x,
                                                              original_multipolygon.centroid.y],
                                                             filename)

    save_polygon_to_disk(directory_output, filename, original_multipolygon, crs_name, properties_dict)


def create_volume_polygons_with_dsm_values(directory_output: str, original_multipolygon: MultiPolygon,
                                           dtm_data: np.ndarray[float], dsm_data: np.ndarray[float],
                                           no_data_dtm: float, no_data_dsm: float,
                                           original_multipolygon_mask: np.ndarray[bool],
                                           original_exterior_mask: np.ndarray[bool],
                                           pixel_size_x: float, pixel_size_y: float, i: int,
                                           crs_name: str, sand_density: float):
    """
        Create volume polygons with DSM values for a given original multipolygon. This method calculates the volume
        based on the DSM values of the original (predicted) multipolygon. The base height of the resulting polygon
        is determined by the DTM height of the exterior of the original multipolygon. The volume is calculated
        for values above and below the base height.

        Parameters:
            directory_output (str): The directory path to save the output files.
            original_multipolygon (MultiPolygon): The original multipolygon representing the predicted sand heap.
            dtm_data (np.ndarray[float]): The Digital Terrain Model (DTM) data.
            dsm_data (np.ndarray[float]): The Digital Surface Model (DSM) data.
            no_data_dtm (float): The no-data value for the DTM data.
            no_data_dsm (float): The no-data value for the DSM data.
            original_multipolygon_mask (np.ndarray[bool]): The mask representing the original multipolygon area.
            original_exterior_mask (np.ndarray[bool]): The mask representing the exterior polygon of the original
                multipolygon (model prediction). Used to calculate the base height of the resulting polygon.
            pixel_size_x (float): The pixel size in the x-direction.
            pixel_size_y (float): The pixel size in the y-direction.
            i (int): The index of the original multipolygon.
            crs_name (str): The name of the Coordinate Reference System (CRS).
            sand_density (float): The density of the sand material.

        Returns:
            None
        """
    result_img_mask = np.zeros_like(dsm_data, dtype=np.uint8)
    result_img_mask[original_multipolygon_mask] = 1

    dsm_data_result = dsm_data[result_img_mask == 1]

    min_dtm = calculate_base_height_with_exterior_polygon(dtm_data, no_data_dtm, original_exterior_mask)

    volume_dsm_above, volume_dsm_below = calculate_volume_above_and_below(dsm_data_result, no_data_dsm, min_dtm,
                                                                          pixel_size_x, pixel_size_y)

    total_volume_t_above = volume_dsm_above * sand_density
    total_volume_t_below = volume_dsm_below * sand_density

    filename = (f"Multipolygon{i}_a{round(volume_dsm_above, 2)}m3_a{round(total_volume_t_above, 2)}t_"
                f"b{round(volume_dsm_below, 2)}m3_b{round(total_volume_t_below, 2)}t.geojson")

    properties_dict = create_properties_dict_volume_polygons(volume_dsm_above, total_volume_t_above,
                                                             volume_dsm_below, total_volume_t_below,
                                                             [original_multipolygon.centroid.x,
                                                              original_multipolygon.centroid.y],
                                                             filename)

    save_polygon_to_disk(directory_output, filename, original_multipolygon, crs_name, properties_dict)


def calculate_volume_above_and_below(dem_data: np.ndarray[float], no_data_dem: float, min_dtm: float,
                                     pixel_size_x: float, pixel_size_y: float) -> Tuple[float, float]:
    """
    Calculate the volume above and below a given minimum DTM value.

    Parameters:
        dem_data (np.ndarray[float]): The Digital Elevation Model (DEM) data.
        no_data_dem (float): The no-data value for the DEM data.
        min_dtm (float): The minimum DTM value.
        pixel_size_x (float): The pixel size in the x-direction.
        pixel_size_y (float): The pixel size in the y-direction.

    Returns: A tuple with the volume above and below the given minimum DTM value.
    """
    mask_nodata_dem = np.ma.masked_values(dem_data, no_data_dem)
    inverted_no_data_mask = ~mask_nodata_dem.mask
    sum_mask_dem_above = inverted_no_data_mask & (dem_data > min_dtm)
    sum_height_dem_above = np.sum(dem_data[sum_mask_dem_above])

    sum_mask_dem_below = inverted_no_data_mask & (dem_data <= min_dtm)
    sum_height_dem_below = np.sum(dem_data[sum_mask_dem_below])

    pixels_dem_above = np.count_nonzero(sum_mask_dem_above)
    pixels_dem_below = np.count_nonzero(sum_mask_dem_below)

    total_height_dem_above = sum_height_dem_above - (min_dtm * pixels_dem_above)
    total_height_dem_below = (min_dtm * pixels_dem_below) - sum_height_dem_below

    volume_dem_above = pixel_size_x * pixel_size_y * total_height_dem_above
    volume_dem_below = pixel_size_x * pixel_size_y * total_height_dem_below

    return volume_dem_above, volume_dem_below


def create_properties_dict_volume_polygons(volume_above_m3, volume_above_t, volume_below_m3, volume_below_t,
                                           centroid_as_list_xy, filename):
    return {"properties": {
        "volume_above_m3": volume_above_m3,
        "volume_above_t": volume_above_t,
        "volume_below_m3": volume_below_m3,
        "volume_below_t": volume_below_t,
        "centroid": centroid_as_list_xy,
        "name": filename.rsplit(".", 1)[0]
    }}


def create_multipolygon_mask(dtm_data: np.ndarray[float], transform: Affine, multipolygon: MultiPolygon) -> np.ndarray[
    bool]:
    """
    Create a mask that contains each subpolygon of a multipolygon.

    Parameters:
        dtm_data (np.ndarray[float]): The Digital Terrain Model (DTM) data grid.
        transform (Affine): The affine transformation representing the spatial relationship between pixel coordinates
            and real-world coordinates.
        multipolygon (MultiPolygon): The MultiPolygon geometry for which to create the mask.

    Returns:
        np.ndarray[bool]: A binary mask where True indicates the area covered by the MultiPolygon.
    """
    multipolygon_mask = np.zeros_like(dtm_data, dtype=np.bool_)
    for subpolygon in multipolygon.geoms:
        multipolygon_mask = np.logical_or(multipolygon_mask, geometry_mask([subpolygon],
                                                                           out_shape=dtm_data.shape,
                                                                           transform=transform,
                                                                           invert=True))
    return multipolygon_mask


def create_height_difference_mask(dsm_data: np.ndarray[float], dtm_data: np.ndarray[float], height_threshold: float) -> \
        np.ndarray[bool]:
    """
    Create a mask indicating areas where the height difference between DSM (Digital Surface Model)
    and DTM (Digital Terrain Model) is above a given threshold. This is used to filter i.e. conveyor belts
    above the sand heaps.

    Parameters:
        dsm_data (np.ndarray[float]): The Digital Surface Model (DSM) data grid.
        dtm_data (np.ndarray[float]): The Digital Terrain Model (DTM) data grid.
        height_threshold (float): The threshold for height difference between DSM and DTM.

    Returns:
        np.ndarray[bool]: A binary mask where True indicates areas where the height difference between DSM and DTM
        is above the threshold.
    """
    height_difference = dsm_data - dtm_data

    return height_difference > height_threshold


def calculate_base_height_with_exterior_polygon(dtm_data: np.ndarray[float], no_data_dem: float,
                                                original_exterior_mask: np.ndarray[bool]) -> float:
    """
    Calculate the base height using the exterior polygons of the original multipolygon.

    Parameters:
        dtm_data (np.ndarray[float]): The Digital Terrain Model (DTM) data grid.
        no_data_dem (float): Value representing no data in the DTM.
        original_exterior_mask (np.ndarray[bool]): The mask representing the exterior polygon of the original
                multipolygon (model prediction). Used to calculate the base height of the resulting polygon.

    Returns:
        float: The average height of the DTM data within the exterior polygon of the convex hull.
    """
    mask_nodata = np.ma.masked_values(dtm_data, no_data_dem)
    inverted_no_data_mask = ~mask_nodata.mask
    valid_dtm_mask = inverted_no_data_mask & original_exterior_mask

    return np.average(dtm_data[valid_dtm_mask])


# bbox is xyxy
def remove_small_box_in_large_box(area_threshold: float, bboxes: torch.Tensor, masks: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Remove smaller bounding boxes within larger bounding boxes based on a given area threshold.

        Args:
            area_threshold (float): Threshold for the intersection area as a ratio of the smaller box's area.
            bboxes (torch.Tensor): Bounding boxes represented as a torch.float32 tensor with shape `(N, 4)` where `N`
            is the number of bounding boxes. The format of the bounding boxes is XYXY (min_x, min_y, max_x, max_y).
            masks (torch.Tensor): Associated masks for the bounding boxes in (N, H, W) torch.uint8 format.

        Returns: Tuple containing the bounding boxes and the masks after removing the smaller boxes and masks.
        """
    number_boxes: int
    number_boxes = bboxes.shape[0]
    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)

    remove_indices = []
    for i in range(number_boxes):
        for j in range(number_boxes):
            if i != j:
                inter_area = find_intersection_area(bboxes[i], bboxes[j])

                if inter_area == 0:
                    continue

                smaller_index = i if areas[i] < areas[j] else j
                smaller_box_area = areas[smaller_index].item()
                # Check if the intersection area is at least area_threshold of the smaller box's area
                if inter_area > area_threshold * smaller_box_area:
                    remove_indices.append(smaller_index)

    # nothing removed
    if len(remove_indices) == 0:
        return bboxes, masks

    removal_mask = torch.ones(number_boxes, dtype=torch.bool)
    removal_mask[remove_indices] = False
    return bboxes[removal_mask], masks[removal_mask]


def find_intersection_area(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
        Calculate the intersection area between two bounding boxes.

        Args:
            box1 (torch.Tensor): A tensor representing the first bounding box in the format [x1, y1, x2, y2],
                where (x1, y1) are the coordinates of the top-left corner and (x2, y2) are the coordinates
                of the bottom-right corner.
            box2 (torch.Tensor): A tensor representing the second bounding box in the same format as box1.

        Returns:
            float: The area of intersection between the two bounding boxes. If the boxes do not intersect,
                returns 0.

        Note:
            This function assumes the input tensors are in the format [x1, y1, x2, y2], where (x1, y1) are the
            coordinates of the top-left corner and (x2, y2) are the coordinates of the bottom-right corner.
        """
    x1 = torch.max(box1[0], box2[0]).item()
    y1 = torch.max(box1[1], box2[1]).item()
    x2 = torch.min(box1[2], box2[2]).item()
    y2 = torch.min(box1[3], box2[3]).item()

    inter_width = max(0, x2 - x1 + 1)
    inter_height = max(0, y2 - y1 + 1)

    return inter_width * inter_height


def merge_patches_with_overlap_top_or_left(result_array: np.ndarray, x: int, y: int, patch_size_height: int,
                                           patch_size_width: int, overlap_direction: str, overlap: int,
                                           processed_mask_tensor: torch.Tensor, iou_threshold: float,
                                           cur_global_id: int) -> int:
    """
        Merge the model prediction patches by considering the overlapping region in the specified direction
        (top or left).

        Args:
            result_array (np.ndarray): The array representing the result where patches will be merged.
            x (int): The x-coordinate of the top-left corner of the patch.
            y (int): The y-coordinate of the top-left corner of the patch.
            patch_size_height (int): The height of the patch.
            patch_size_width (int): The width of the patch.
            overlap_direction (str): The direction of overlap, either 'left' or 'top'.
            overlap (int): The size of the overlap.
            processed_mask_tensor (torch.Tensor): Mask tensor of [N, H, W] torch.uint8 format.
            iou_threshold (float): The IoU threshold for merging predicted instances with existing ones
                in the overlapping region.
            cur_global_id (int): The current global ID for instance masks. The global ID represents the number
                of instances on the image (including the background).

        Returns:
            int: The updated current global ID after merging.
        """
    if overlap_direction == 'left':
        overlap_region = torch.from_numpy(result_array[y:y + patch_size_height, x:x + overlap])
    elif overlap_direction == 'top':
        overlap_region = torch.from_numpy(result_array[y:y + overlap, x:x + patch_size_width])
    else:
        raise ValueError("Invalid overlap direction. Use 'left' or 'top'.")

    # get every number (instance) in the overlapping area that is not the background
    unique_numbers = torch.unique(overlap_region[overlap_region != 0])

    if unique_numbers.numel() == 0:
        # only background in overlapping region until now, so copy model predictions
        for predicted_instance_mask in processed_mask_tensor:
            cur_global_id = create_new_instance_in_result_array(cur_global_id, result_array, patch_size_height,
                                                                patch_size_width, x, y, predicted_instance_mask)
        return cur_global_id

    # get a boolean tensor for each unique number (existing instance) in the overlapping area
    existing_instances_in_overlap_region = create_existing_instances_bool_tensor_in_overlap_area(unique_numbers,
                                                                                                 overlap_region)

    # check if masks (predicted instances) in overlapping region have iou above threshold
    # with existing instances
    for predicted_instance_mask in processed_mask_tensor:
        if overlap_direction == 'left':
            predicted_instance_mask_overlap_region = predicted_instance_mask[0: patch_size_height, 0:overlap]
        else:
            predicted_instance_mask_overlap_region = predicted_instance_mask[0: overlap, 0:patch_size_width]

        has_merged = try_to_merge_predicted_and_existing_instances(predicted_instance_mask_overlap_region,
                                                                   existing_instances_in_overlap_region,
                                                                   iou_threshold, unique_numbers,
                                                                   result_array, x, y, predicted_instance_mask,
                                                                   patch_size_height, patch_size_width)

        if not has_merged:
            cur_global_id = create_new_instance_in_result_array(cur_global_id, result_array, patch_size_height,
                                                                patch_size_width, x, y, predicted_instance_mask)

    return cur_global_id


def merge_patches_with_overlap_top_and_left(result_array: np.ndarray, x: int, y: int, patch_size_height: int,
                                            patch_size_width: int, overlap: int,
                                            processed_mask_tensor: torch.Tensor,
                                            iou_threshold: float, cur_global_id: int) -> int:
    """
    Merge the model prediction patches by considering the overlapping region in the specified direction
        (top and left).

    Args:
        result_array (np.ndarray): The array representing the result where patches will be merged.
        x (int): The x-coordinate of the top-left corner of the patch.
        y (int): The y-coordinate of the top-left corner of the patch.
        patch_size_height (int): The height of the patch.
        patch_size_width (int): The width of the patch.
        overlap (int): The size of the overlap.
        processed_mask_tensor (torch.Tensor): Mask tensor of [N, H, W] torch.uint8 format.
        iou_threshold (float): The IoU threshold for merging instance masks.
        cur_global_id (int): The current global ID for instance masks. The global ID represents the number
                of instances on the image (including the background).

    Returns:
        int: The updated current global ID after merging.
    """
    overlap_top_region = torch.from_numpy(result_array[y:y + overlap, x:x + patch_size_width])
    overlap_left_region = torch.from_numpy(result_array[y:y + patch_size_height,
                                           x:x + overlap])

    # get every number (instance) in the overlapping area that is not the background
    unique_numbers_left = torch.unique(overlap_left_region[overlap_left_region != 0])
    unique_numbers_top = torch.unique(overlap_top_region[overlap_top_region != 0])

    if unique_numbers_left.numel() == 0 and unique_numbers_top.numel() == 0:
        # only background in overlapping region until now
        for predicted_instance_mask in processed_mask_tensor:
            cur_global_id = create_new_instance_in_result_array(cur_global_id, result_array, patch_size_height,
                                                                patch_size_width, x, y, predicted_instance_mask)
        return cur_global_id
    elif unique_numbers_left.numel() == 0:
        # only check top for overlap
        return merge_patches_with_overlap_top_or_left(result_array=result_array, x=x, y=y,
                                                      patch_size_height=patch_size_height,
                                                      patch_size_width=patch_size_width,
                                                      overlap=overlap,
                                                      overlap_direction='top',
                                                      processed_mask_tensor=processed_mask_tensor,
                                                      iou_threshold=iou_threshold,
                                                      cur_global_id=cur_global_id)

    elif unique_numbers_top.numel() == 0:
        # only check left for overlap
        return merge_patches_with_overlap_top_or_left(result_array=result_array, x=x, y=y,
                                                      patch_size_height=patch_size_height,
                                                      patch_size_width=patch_size_width,
                                                      overlap=overlap,
                                                      overlap_direction='left',
                                                      processed_mask_tensor=processed_mask_tensor,
                                                      iou_threshold=iou_threshold,
                                                      cur_global_id=cur_global_id)
    else:
        # get existing instances in both regions
        existing_instances_in_overlap_left = create_existing_instances_bool_tensor_in_overlap_area(unique_numbers_left,
                                                                                                   overlap_left_region)

        existing_instances_in_overlap_top = create_existing_instances_bool_tensor_in_overlap_area(unique_numbers_top,
                                                                                                  overlap_top_region)

        for predicted_instance_mask in processed_mask_tensor:
            predicted_instance_mask_overlap_top = predicted_instance_mask[0: overlap, 0:patch_size_width]
            predicted_instance_mask_overlap_left = predicted_instance_mask[0: patch_size_height, 0:overlap]

            # if we have overlap to the left,
            # we always take the prediction from the left and the instance from the left
            # since we cannot assign two instances to one predicted mask anyway

            has_merged = try_to_merge_predicted_and_existing_instances(predicted_instance_mask_overlap_left,
                                                                       existing_instances_in_overlap_left,
                                                                       iou_threshold, unique_numbers_left,
                                                                       result_array, x, y, predicted_instance_mask,
                                                                       patch_size_height, patch_size_width)
            if not has_merged:
                # if we have no overlap to the left, check for overlap to the top
                has_merged = try_to_merge_predicted_and_existing_instances(predicted_instance_mask_overlap_top,
                                                                           existing_instances_in_overlap_top,
                                                                           iou_threshold, unique_numbers_top,
                                                                           result_array, x, y, predicted_instance_mask,
                                                                           patch_size_height, patch_size_width)
            if not has_merged:
                # if we still have not merged, we have no match in either region,
                # and create a new instance
                cur_global_id = create_new_instance_in_result_array(cur_global_id, result_array, patch_size_height,
                                                                    patch_size_width, x, y, predicted_instance_mask)

        return cur_global_id


def create_new_instance_in_result_array(cur_global_id: int, result_array: np.ndarray, patch_size_height: int,
                                        patch_size_width: int, x: int, y: int,
                                        predicted_instance_mask: torch.Tensor) -> int:
    """
    Create a new instance in the result_array at the specified location with the predicted instance mask.

    Args:
        cur_global_id (int): The current global ID for instance masks. The global ID represents the number
                of instances on the image (including the background).
        result_array (np.ndarray): The array representing the result where the new instance will be created.
        patch_size_height (int): The height of the patch.
        patch_size_width (int): The width of the patch.
        x (int): The x-coordinate of the top-left corner of the patch.
        y (int): The y-coordinate of the top-left corner of the patch.
        predicted_instance_mask (torch.Tensor): The predicted instance mask to create a new instance
            in the result array.

    Returns:
        int: The updated current global ID after creating the new instance.
    """
    instance_mask = predicted_instance_mask == 1
    result_array[y:y + patch_size_height, x:x + patch_size_width][instance_mask] = cur_global_id
    cur_global_id += 1
    return cur_global_id


def extend_existing_instance_in_result_array(predicted_instance_mask: torch.Tensor, result_array: np.ndarray,
                                             patch_size_height: int, patch_size_width: int, x: int, y: int,
                                             number: int):
    """
    Extend an existing instance in the result_array with the predicted instance mask.

    Args:
        predicted_instance_mask (torch.Tensor): The predicted instance mask.
        result_array (np.ndarray): The array representing the result where the instance will be extended.
        patch_size_height (int): The height of the patch.
        patch_size_width (int): The width of the patch.
        x (int): The x-coordinate of the top-left corner of the patch.
        y (int): The y-coordinate of the top-left corner of the patch.
        number (int): The number of the existing instance.
    Returns:
        None
    """
    instance_mask = predicted_instance_mask == 1
    result_array[y:y + patch_size_height, x:x + patch_size_width][instance_mask] = number


def create_existing_instances_bool_tensor_in_overlap_area(unique_numbers: torch.Tensor,
                                                          overlap_region: torch.Tensor) -> torch.Tensor:
    """
    Create boolean tensors for each unique instance (number) in the overlap area.

    Args:
        unique_numbers (torch.Tensor): Unique numbers representing instances in the overlap area.
        overlap_region (torch.Tensor): The overlap region in the result array.

    Returns:
        torch.Tensor: Stack of boolean tensors for each unique number.
    """
    boolean_tensors_list = []
    for number in unique_numbers:
        boolean_tensor = (overlap_region == number)
        boolean_tensor: torch.Tensor
        boolean_tensors_list.append(boolean_tensor)

    return torch.stack(boolean_tensors_list)


def try_to_merge_predicted_and_existing_instances(predicted_instance_mask_overlap_region: torch.Tensor,
                                                  existing_instances_in_overlap_region: torch.Tensor,
                                                  iou_threshold: float, unique_numbers: torch.Tensor,
                                                  result_array: np.ndarray, x: int, y: int,
                                                  predicted_instance_mask: torch.Tensor,
                                                  patch_size_height: int, patch_size_width: int) -> bool:
    """
    Try to merge predicted and existing instances in the overlap region.

    Args:
        predicted_instance_mask_overlap_region (torch.Tensor): Predicted instance mask in the overlap region.
        existing_instances_in_overlap_region (torch.Tensor): Existing instance masks in the overlap region.
        iou_threshold (float): IoU threshold for merging instance masks.
        unique_numbers (torch.Tensor): Unique numbers representing existing instances in the overlap region.
        result_array (np.ndarray): The array representing the result where instances could be merged.
        x (int): The x-coordinate of the top-left corner of the patch.
        y (int): The y-coordinate of the top-left corner of the patch.
        predicted_instance_mask (torch.Tensor): The predicted instance mask. Needed since we need to
            extend in the overlapping area AND in the remaining patch area when we merge.
        patch_size_height (int): The height of the patch.
        patch_size_width (int): The width of the patch.

    Returns:
        bool: True if merging is successful, False otherwise.
    """
    for (i, existing_instance_mask_overlap_region) in enumerate(existing_instances_in_overlap_region):
        iou = compute_binary_mask_iou(existing_instance_mask_overlap_region, predicted_instance_mask_overlap_region)
        if iou > iou_threshold:
            extend_existing_instance_in_result_array(predicted_instance_mask, result_array, patch_size_height,
                                                     patch_size_width, x, y, unique_numbers[i].item())
            return True
    return False


def load_final_multipolygons_and_geojson(path_polygons) -> Tuple[List[MultiPolygon], List[Dict[str, Any]]]:
    """
        Load multipolygons and associated GeoJSON data from GeoJSON files in a directory.

        Args:
            path_polygons (str): Directory path containing GeoJSON files.

        Returns:
            Tuple[List[MultiPolygon], List[Dict[str, Any]]]: A tuple containing a List of Shapely MultiPolygon objects
                representing the loaded multipolygons and a List of dictionaries representing the GeoJSON data
                of these multipolygons.
    """
    multipolygon_files = [os.path.join(path_polygons, f) for f in
                          os.listdir(path_polygons) if
                          f.endswith('.geojson')]

    multipolygons = []
    geojson_list = []

    for multipolygon_file in multipolygon_files:
        with open(multipolygon_file, 'r') as poly_source:
            geojson_data = json.load(poly_source)
            geojson_list.append(geojson_data)
            multipolygons.append(shape(geojson_data['features'][0]['geometry']))

    return multipolygons, geojson_list


def transform_epsg32632_geometry_to_wgs84(geometry: Union[MultiPolygon, Polygon, Point]) -> Union[
    MultiPolygon, Polygon, Point]:
    """
        Transform a geometry (MultiPolygon, Polygon, Point) from EPSG:32632 (UTM Zone 32N) to WGS84.

        Args:
            geometry (Union[MultiPolygon, Polygon, Point]): Shapely geometry object in EPSG:32632 coordinates.

        Returns:
            Union[MultiPolygon, Polygon, Point]: Transformed Shapely geometry object in WGS84 coordinates.
    """
    wgs84_transform = pyproj.CRS("wgs84")
    epsg32632_transform = pyproj.CRS("EPSG:32632")
    projection = pyproj.Transformer.from_crs(epsg32632_transform, wgs84_transform, always_xy=True).transform

    geometry_shapely = shape(geometry)

    geometry_transformed = transform(projection, geometry_shapely)

    return geometry_transformed


if __name__ == '__main__':
    print("")
