import rioxarray
import xarray
import rasterio
import argparse


def stack(input_paths, output_path):
    tifs = []
    bands_off = 0
    for path in input_paths:
        tif = rioxarray.open_rasterio(path, masked=True)
        tif = tif.assign_coords(band=tif.coords["band"] + bands_off)
        bands_off += len(tif.coords["band"])
        tifs.append(tif)

    ref = tifs[0]
    for tif_idx, tif in enumerate(tifs[1:]):
        tifs[tif_idx + 1] = tif.rio.reproject_match(ref)

    stacked = xarray.concat(tifs, dim="band")
    
    count, height, width = stacked.shape
    meta = dict(
        driver="GTiff", dtype=rasterio.float32, nodata=ref.rio.nodata,
        width=width, height=height, count=count, crs=ref.rio.crs,
        transform=ref.rio.transform()
    )
    stack = rasterio.open(output_path, "w", **meta)
    
    for band_idx in range(count):
        band = stacked[band_idx]
        stack.write(band, band_idx + 1)
    
    stack.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", help="Path to output .tif file")
    parser.add_argument("input_paths", nargs="*", help="Paths to .tif files for merging")
    args = parser.parse_args()
    
    stack(args.input_paths, args.output_path)


if __name__ == "__main__":
    main()
