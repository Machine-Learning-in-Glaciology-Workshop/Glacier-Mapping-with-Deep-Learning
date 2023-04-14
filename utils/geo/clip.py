import rasterio
import rasterio.mask
import fiona
import argparse


def clip(input_path, mask_path, output_path):
    with fiona.open(mask_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(input_path) as src:
        clipped, trans = rasterio.mask.mask(src, shapes, crop=True)
        meta = src.meta

    meta.update(
        {
            "driver": "GTiff",
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": trans
        }
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(clipped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to .tif file to clip")
    parser.add_argument("mask_path", help="Path to .shp file to use as mask")
    parser.add_argument("output_path", help="Path to output .tif file")
    args = parser.parse_args()

    clip(args.input_path, args.mask_path, args.output_path)


if __name__ == "__main__":
    main()
