import rasterio
import rasterio.features
import fiona
import argparse


def burn(input_path, output_path, reference_path):
    with fiona.open(input_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(reference_path) as ref:
        rasterized = rasterio.features.rasterize(
            [(geometry, 1) for geometry in shapes],
            out_shape=ref.shape,
            transform=ref.transform
        )
        meta = ref.meta

    meta.update(
        {
            "count": 1,
            "dtype": rasterio.uint8,
            "nodata": 0
        }
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(rasterized, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to .shp file to burn")
    parser.add_argument("output_path", help="Path to output .tif file")
    parser.add_argument("reference_path", help="Path to reference .tif file")
    args = parser.parse_args()

    burn(args.input_path, args.output_path, args.reference_path)


if __name__ == "__main__":
    main()
