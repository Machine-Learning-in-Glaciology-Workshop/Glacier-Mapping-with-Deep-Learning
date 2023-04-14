from osgeo import gdal
import rasterio
import pyproj
import utils
import argparse


def slope(input_path, output_path):
    with rasterio.open(input_path, "r") as src:
        transformer = pyproj.Transformer.from_crs(src.crs, "4326")
        xmin, ymin, xmax, ymax = src.bounds
        xavg, yavg = (xmin + xmax) / 2, (ymin + ymax) / 2
        _, lat = transformer.transform([xavg], [yavg])
        lat = lat[0]
    z_factor = utils.get_z_factor(lat)
    gdal.DEMProcessing(
        output_path, input_path, "slope", 
        options=gdal.DEMProcessingOptions(scale=1/z_factor)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to .tif DEM file in meters")
    parser.add_argument("output_path", help="Path to output .tif file")
    args = parser.parse_args()

    slope(args.input_path, args.output_path)
