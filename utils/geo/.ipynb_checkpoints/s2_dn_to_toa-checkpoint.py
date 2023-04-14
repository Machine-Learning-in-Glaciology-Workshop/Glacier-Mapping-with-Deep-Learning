import rasterio
import numpy as np
import argparse


def dn_to_toa(input_path, output_path):
    with rasterio.open(input_path) as src:
        array = src.read()
        meta = src.meta

    array = array / 1e4
    meta.update({"dtype": np.float32})

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to .tif file to clip")
    parser.add_argument("--output_path", help="Path to output .tif file")
    args = parser.parse_args()

    dn_to_toa(args.input_path, args.output_path if args.output_path else args.input_path)


if __name__ == "__main__":
    main()
