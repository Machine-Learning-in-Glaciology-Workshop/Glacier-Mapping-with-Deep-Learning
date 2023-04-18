import rioxarray
import rasterio
import rasterio.windows
import numpy as np
import dask
import utils
import argparse


def mosaic(
    input_paths, output_path, window_size=2048, chunk_size=512, 
    num_workers=16, scheduler="threads", nodata=None
):
    tifs = []
    original_bounds = []
    for path in input_paths:
        tif = rioxarray.open_rasterio(path, masked=True, chunks={"x": chunk_size, "y": chunk_size})
        tifs.append(tif)
        original_bounds.append(tif.rio.bounds())

    left, bottom, right, top = original_bounds[0]
    for bounds in original_bounds[1:]:
        left = min(left, bounds[0])
        bottom = min(bottom, bounds[1])
        right = max(right, bounds[2])
        top = max(top, bounds[3])

    for tif_idx, tif in enumerate(tifs):
        tifs[tif_idx] = tif.rio.pad_box(left, bottom, right, top).chunk({"x": chunk_size, "y": chunk_size})
    ref = tifs[0]
    
    count, height, width = ref.shape
    meta = dict(
        driver="GTiff", dtype=rasterio.float32, nodata=nodata,
        width=width, height=height, count=count, crs=ref.rio.crs,
        transform=ref.rio.transform()
    )
    mosaic = rasterio.open(output_path, "w", **meta)
    
    row_off = 0
    while row_off < height:
        col_off = 0
        while col_off < width:
            y_slice = slice(row_off, min(row_off + window_size, height))
            x_slice = slice(col_off, min(col_off + window_size, width))
            write_window = rasterio.windows.Window.from_slices(y_slice, x_slice)
            write_window_bounds = rasterio.windows.bounds(write_window, ref.rio.transform())
            
            windows = []
            for tif, bounds in zip(tifs, original_bounds):
                if not utils.are_bounds_intersect(bounds, write_window_bounds):
                    continue
                window = tif.isel(y=y_slice, x=x_slice)
                if windows and window.shape != windows[0].shape:
                    window = window.rio.reproject_match(windows[0])
                if nodata is not None:
                    window = window.where(window != nodata, np.nan)
                windows.append(window)
            
            if not windows:
                col_off += window_size
                continue
            
            stack = dask.array.stack(windows)
            median = dask.array.nanmedian(stack, axis=0)
            median.persist(scheduler=scheduler, num_workers=num_workers)
            
            mosaic.write(median, window=write_window)

            col_off += window_size
        row_off += window_size
        
    mosaic.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", help="Path to output .tif file")
    parser.add_argument("input_paths", nargs="*", help="Paths to .tif files for mosaicing")
    parser.add_argument("--window_size", default=2048, type=int, help="Size of temporary windows to process data")
    parser.add_argument("--chunk_size", default=512, type=int, help="Chunk size for parallel processing")
    parser.add_argument("--num_workers", default=16, type=int, help="Number of workers for parallel processing")
    parser.add_argument("--scheduler", default="threads", type=str, help="Scheduler for parallel processing")
    parser.add_argument("--nodata", type=float, default=None, help="Nodata value")
    args = parser.parse_args()
    
    mosaic(
        args.input_paths, args.output_path, args.window_size, args.chunk_size, 
        args.num_workers, args.scheduler, args.nodata
    )


if __name__ == "__main__":
    main()
