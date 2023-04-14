import nicesnappy
from nicesnappy import Read, Write, Chain, SNAPOperator
import geopandas
import argparse
import os


def preprocess_s1(input_path, mask_path, output_path):
    apply_orbit_file = SNAPOperator("Apply-Orbit-File")
    thermal_noise_removal = SNAPOperator("ThermalNoiseRemoval")
    border_noise_removal = SNAPOperator("Remove-GRD-Border-Noise")
    calibration = SNAPOperator("Calibration", {
        "selectedPolarisations": "VV"
    })
    speckle_filter = SNAPOperator("Speckle-Filter")
    terrain_correction = SNAPOperator("Terrain-Correction", {
        "demName": "Copernicus 30m Global DEM",
        "pixelSpacingInMeter": "10"
    })
    region_shp = geopandas.read_file(mask_path)
    region_shp = region_shp.explode()
    wkt = region_shp.iloc[0].geometry.wkt
    aoi_subset = SNAPOperator("Subset", {
        "geoRegion": wkt
    })

    graph = Chain([
        Read(input_path),
        apply_orbit_file,
        thermal_noise_removal,
        border_noise_removal,
        calibration,
        speckle_filter,
        terrain_correction,
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
    preprocess_s1(args.input_path, args.mask_path, args.output_path)


if __name__ == "__main__":
    main()
