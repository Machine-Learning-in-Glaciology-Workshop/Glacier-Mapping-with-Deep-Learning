import rioxarray
import argparse


def reproject(
    input_path, output_path, epsg, chunk_size=512, num_workers=16, 
    scheduler="threads"
):
    src = rioxarray.open_rasterio(input_path, chunks={"x": chunk_size, "y": chunk_size})
    dst = src.rio.reproject(f"EPSG:{epsg}")
    dst.rio.to_raster(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to .tif file to clip")
    parser.add_argument("epsg", help="EPSG code of new CRS")
    parser.add_argument("--output_path", help="Path to output .tif file")
    parser.add_argument("--chunk_size", default=512, type=int, help="Chunk size for parallel processing")
    parser.add_argument("--num_workers", default=16, type=int, help="Number of workers for parallel processing")
    parser.add_argument("--scheduler", default="threads", type=str, help="Scheduler for parallel processing")
    args = parser.parse_args()

    reproject(
        args.input_path, args.output_path if args.output_path else args.input_path, args.epsg,
        args.chunk_size, args.num_workers, args.scheduler
    )


if __name__ == "__main__":
    main()
