import geopandas as gpd
import glob
import os
import pandas as pd

# Path to your folder with shapefiles
# <-- replace with your actual path
folder_path = "/Users/jovanaknezevic/Documents/cci_workshop/workshop_shapefiles"

# Get all .shp files in that folder
shapefiles = glob.glob(os.path.join(folder_path, "*.shp"))

# Define the target CRS (WGS84)
target_crs = "EPSG:4326"

# Read, reproject, and store all GeoDataFrames
gdf_list = [gpd.read_file(shp).to_crs(target_crs) for shp in shapefiles]

# Concatenate into one GeoDataFrame
merged_gdf = gpd.GeoDataFrame(
    pd.concat(gdf_list, ignore_index=True), crs=target_crs)

# Save merged shapefile
output_path = os.path.join(folder_path, "merged_output.shp")
merged_gdf.to_file(output_path)

print(f"Merged shapefile saved to: {output_path}")
