import nicesnappy
from nicesnappy import Read, Write, Chain, SNAPOperator
import geopandas
import argparse
import os


def preprocess_s2(input_path, mask_path, output_path):
    resampling = SNAPOperator("S2Resampling", {
        "resolution": "10",
        "upsampling": "Nearest"
    })
    bands_subset = SNAPOperator("Subset", {
        "sourceBands": "B2,B3,B4,B8,B11,B12"
    })
    region_shp = geopandas.read_file(mask_path)
    region_shp = region_shp.explode()
    wkt = region_shp.iloc[0].geometry.wkt
    aoi_subset = SNAPOperator("Subset", {
        "geoRegion": wkt
    })

    graph = Chain([
        Read(input_path),
        resampling,
        bands_subset,
        aoi_subset,
        Write(output_path, "GeoTIFF-BigTIFF")
    ])
    graph.apply()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to input file")
    parser.add_argument("mask_path", help="Path to .shp file to use as mask")
    parser.add_argument("output_path", help="Path to output .tif file")
    args = parser.parse_args()

    nicesnappy.initialize(os.path.join("/home", "eouser", ".snap", "snap-python"))
    preprocess_s2(args.input_path, args.mask_path, args.output_path)


if __name__ == "__main__":
    main()
