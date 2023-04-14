import rasterio
import rasterio.windows
import numpy as np
import argparse
import os
import glob
import gc


def mosaic(input_paths, output_path, nodata=0):
    tifs = []
    for path in input_paths:
        tif = rasterio.open(path, "r+")
        tif.nodata = nodata
        tifs.append(tif)

    left, bottom, right, top = tifs[0].bounds
    for tif in tifs[1:]:
        bounds = tif.bounds
        left = min(left, bounds.left)
        bottom = min(bottom, bounds.bottom)
        right = max(right, bounds.right)
        top = max(top, bounds.top)

    arrays = []
    for tif in tifs:
        window = rasterio.windows.from_bounds(
            left=left, bottom=bottom, right=right, top=top,
            transform=tif.transform,
        )
        array = tif.read(
            boundless=True,
            window=window,
            fill_value=nodata,
            out_shape=None if not arrays else arrays[0].shape,
            resampling=rasterio.enums.Resampling.nearest,
            masked=False
        )
        array[array==nodata] = np.nan
        arrays.append(array)

    mosaic = np.nanmedian(arrays, axis=0)
    del arrays
    gc.collect()

    meta = tifs[0].meta.copy()
    window = rasterio.windows.from_bounds(
        left=left, bottom=bottom, right=right, top=top,
        transform=tifs[0].transform,
    )
    meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": rasterio.windows.transform(window, tifs[0].transform)
        }
    )

    with rasterio.open(output_path, "w", **meta) as output:
        output.write(mosaic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder with .tif files to mosaic")
    parser.add_argument("--nodata", type=float, default=0, help="Nodata value")
    args = parser.parse_args()

    folder = args.folder
    tif_mask = os.path.join(folder, "*.tif")
    tif_paths = glob.glob(tif_mask)

    output_path = os.path.join(folder, "mosaic.tif")
    mosaic(tif_paths, output_path, args.nodata)


if __name__ == "__main__":
    main()
