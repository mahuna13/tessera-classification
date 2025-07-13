import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from geopandas import gpd
from pyproj import Transformer
from rasterio.merge import merge
from shapely.geometry import box


def stitch_classification_tiles(
    source_dir: str, output_path: str, cleanup: bool = False
):
    """
    Merges (stitches) a directory of classified GeoTIFF tiles into a single map.
    This version builds a clean metadata profile for the output file to ensure compatibility.
    """
    intermediate_files = list(Path(source_dir).glob("*.tif"))
    if not intermediate_files:
        print("No intermediate files found to stitch.")
        return None

    print(f"\nStitching {len(intermediate_files)} tiles into a master mosaic...")
    src_files_to_mosaic = [rasterio.open(path) for path in intermediate_files]

    try:
        # This part remains the same
        mosaic, out_trans = merge(src_files_to_mosaic)

        # --- THIS IS THE CORRECTED LOGIC ---
        # Build a new, clean metadata profile for the output file
        # instead of copying from a source file.
        out_meta = {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": src_files_to_mosaic[
                0
            ].crs,  # All files are merged into the CRS of the first file
            "count": 1,
            "dtype": mosaic.dtype,  # Use the dtype of the actual merged data
            "nodata": src_files_to_mosaic[0].nodata,  # Preserve the nodata value
            "compress": "lzw",
        }
        # ------------------------------------

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        print(f"✅ Successfully created final classification map: {output_path}")
        return output_path

    finally:
        for src in src_files_to_mosaic:
            src.close()
        if cleanup:
            print(f"Cleaning up temporary directory: {source_dir}")
            shutil.rmtree(source_dir)


# --- The Worker Function (expects 6 arguments) ---
def classify_single_tile_worker(args: tuple):
    """
    Worker function executed by each parallel process. It classifies one tile.
    """
    # This unpacks the 6 arguments passed from the main function
    tile_lat, tile_lon, year, model, output_dir, gt = args
    try:
        # Fetch and classify
        embedding_array = gt.fetch_embedding(
            lat=tile_lat, lon=tile_lon, year=year, progressbar=False
        )
        h, w, c = embedding_array.shape
        class_map = model.predict(embedding_array.reshape(-1, c)).reshape(h, w)

        # Convert to a standard, saveable data type
        class_map = class_map.astype(np.uint8)

        # Get georeferencing info
        landmask_path = gt._fetch_landmask(
            lat=tile_lat, lon=tile_lon, progressbar=False
        )
        with rasterio.open(landmask_path) as landmask_src:
            src_crs, src_transform = landmask_src.crs, landmask_src.transform

        # Save the file
        output_path = Path(output_dir) / f"classified_{tile_lat:.2f}_{tile_lon:.2f}.tif"
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=class_map.dtype,
            crs=src_crs,
            transform=src_transform,
            compress="lzw",
        ) as dst:
            dst.write(class_map, 1)

        return str(output_path)
    except Exception as e:
        print(f"  ! Worker for tile ({tile_lat:.2f}, {tile_lon:.2f}) failed: {e}")
        return None


def load_training_labels_from_json(
    file_path: str, source_crs: str = "EPSG:27700"
) -> list:
    """
    Loads labeled points from the specified JSON file and converts them into the
    format needed for analysis.

    Args:
        file_path: The path to the JSON file.
        source_crs: The Coordinate Reference System of the points in the JSON file.
                    Defaults to "EPSG:27700" for the British National Grid.

    Returns:
        A list of dictionaries, with each dictionary representing a labeled point
        with coordinates converted to WGS84 (EPSG:4326).
    """
    # 1. Open and load the JSON data from the file
    with open(file_path, "r") as f:
        data = json.load(f)

    # The points are nested in the 'training_points' key
    raw_points = data["training_points"][0]
    print(len(raw_points))

    # 2. Set up a transformer to convert from the source CRS to WGS84
    # The Tessera grid is based on WGS84 (EPSG:4326), so we must convert our points.
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)

    # 3. Process each point, transform its coordinates, and format it correctly
    processed_points = []
    for item in raw_points:
        # Extract the coordinate and label from the nested list structure
        coord = item[0]
        label = item[1]

        # pyproj expects (x, y) or (Easting, Northing) and returns (lon, lat)
        lon, lat = transformer.transform(coord[0], coord[1])

        processed_points.append(
            {
                "lat": lat,
                "lon": lon,
                "crs": "EPSG:4326",  # The CRS is now WGS84
                "label": label,
            }
        )

    return processed_points


def load_visualization_mappings(
    json_file_path: str,
) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, str]]:
    """
    Loads the necessary color and label mappings from a JSON file for visualization.

    This function does NOT fetch embeddings and is designed to be used after
    a raster has already been classified.

    Args:
        json_file_path: Path to the JSON file containing the labeled points
                        and the class color map.

    Returns:
        A tuple containing:
        - label_to_id (dict): Mapping from string labels to integer IDs.
        - id_to_label (dict): Mapping from integer IDs back to string labels.
        - class_colors (dict): Mapping from string labels to hex color codes.
    """
    print("--- Loading Mappings for Visualization ---")

    # 1. Load the entire JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # 2. Extract the list of all labels from the training points
    # The structure in the JSON is a list containing another list of points
    raw_points = data.get("training_points", [[]])[0]
    all_labels = [item[1] for item in raw_points]

    # 3. Create the label-to-integer mappings based on all unique labels found
    unique_labels = sorted(list(set(all_labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    print("Generated class mappings from all labels in the file.")

    # 4. Load the color map from the JSON file
    class_colors = data.get("class_color_map", {})
    print(f"Loaded {len(class_colors)} color definitions.")

    print("✅ Mappings ready for visualization.")

    return label_to_id, id_to_label, class_colors


def roi_from_points(points, crs="EPSG:4326"):
    # Create a temporary DataFrame to easily find the min/max lat/lon
    points_df = pd.DataFrame(points)
    min_lon, max_lon = points_df["lon"].min(), points_df["lon"].max()
    min_lat, max_lat = points_df["lat"].min(), points_df["lat"].max()

    # Add a small buffer for context around the points (e.g., ~1km)
    buffer = 0.01
    roi_bounds = (
        min_lon - buffer,
        min_lat - buffer,
        max_lon + buffer,
        max_lat + buffer,
    )

    # Create the final GeoDataFrame for the ROI
    roi_poly = box(*roi_bounds)
    return gpd.GeoDataFrame([1], geometry=[roi_poly], crs=crs)
