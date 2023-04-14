import rasterio
import rasterio.merge
import argparse
import os
import glob


def merge(input_paths, output_path, nodata=0, method="last"):
    tifs = []
    for path in input_paths:
        tif = rasterio.open(path, "r+")
        tif.nodata = nodata
        tifs.append(tif)

    merged, trans = rasterio.merge.merge(tifs, method=method)

    meta = tifs[0].meta.copy()
    meta.update(
        {
            "driver": "GTiff",
            "height": merged.shape[1],
            "width": merged.shape[2],
            "transform": trans
        }
    )

    with rasterio.open(output_path, "w", **meta) as output:
        output.write(merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", help="Path to output .tif file")
    parser.add_argument("input_paths", nargs="*", help="Paths to .tif files for merging")
    parser.add_argument("--nodata", type=float, default=0, help="Nodata value")
    parser.add_argument("--method", choices=["first", "last", "min", "max"], default="last", help="Method used to merge files")
    args = parser.parse_args()
    
    merge(args.input_paths, args.output_path, args.nodata, args.method)


if __name__ == "__main__":
    main()
