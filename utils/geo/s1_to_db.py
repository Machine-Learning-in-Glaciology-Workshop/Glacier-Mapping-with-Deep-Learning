import rasterio
import numpy as np
import argparse


def to_db(input_path, output_path, epsilon=1e-6):
    with rasterio.open(input_path) as src:
        array = src.read()
        meta = src.meta

    array = 10 * np.log10(array + epsilon)
    meta.update(
        {
            "dtype": np.float32,
            "nodata": 10 * np.log10(epsilon)
        }
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to .tif file to clip")
    parser.add_argument("--output_path", help="Path to output .tif file")
    parser.add_argument("--epsilon", default=1e-6, type=float, help="Epsilon to avoid NANs in log10")
    args = parser.parse_args()

    to_db(args.input_path, args.output_path if args.output_path else args.input_path, args.epsilon)


if __name__ == "__main__":
    main()
