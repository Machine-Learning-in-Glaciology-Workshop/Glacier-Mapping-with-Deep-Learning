import geopandas
import shapely
import argparse


TILE_SIZE = 10000 # m


def create_tiles(input_path, output_path, region):
    tiles = {
        "row": [],
        "column": [],
        "geometry": []
    }

    bbox = geopandas.read_file(input_path)
    best_utm = bbox.estimate_utm_crs()
    bbox = bbox.to_crs(best_utm)

    x_min, y_min, x_max, y_max = bbox.geometry.total_bounds
    x_centroid, y_centroid = (x_min + x_max) / 2, (y_min + y_max) / 2
    
    num_of_tiles_x = int((x_max - x_min) / TILE_SIZE + 2)
    num_of_tiles_y = int((y_max - y_min) / TILE_SIZE + 2)

    x_start = x_centroid - num_of_tiles_x / 2 * TILE_SIZE
    y_start = y_centroid - num_of_tiles_y / 2 * TILE_SIZE

    tiles = {
        "codename": [],
        "region_codename": [],
        "row": [],
        "column": [],
        "geometry": []
    }

    y, y_count = y_start, 1
    for _ in range(num_of_tiles_y):
        x, x_count = x_start, 1
        for _ in range(num_of_tiles_x):
            tile_codename = f"{region}-{y_count}-{x_count}"
            tile_geometry = shapely.geometry.Polygon(
                zip(
                    [x, x + TILE_SIZE, x + TILE_SIZE, x],
                    [y, y, y + TILE_SIZE, y + TILE_SIZE]
                )
            )

            if tile_geometry.intersects(bbox.iloc[0].geometry):
                tiles["codename"].append(tile_codename)
                tiles["region_codename"].append(region)
                tiles["row"].append(y_count)
                tiles["column"].append(x_count)
                tiles["geometry"].append(tile_geometry)

            x += TILE_SIZE
            x_count += 1
        y += TILE_SIZE
        y_count += 1

    tiles_dataframe = geopandas.GeoDataFrame(tiles, crs=best_utm)
    tiles_dataframe = tiles_dataframe.to_crs("EPSG:4326")
    tiles_dataframe.to_file(output_path)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to input .shp bbox file")
    parser.add_argument("output_path", help="Path to output .shp file")
    parser.add_argument("region", help="Region prefix")
    args = parser.parse_args()

    create_tiles(args.input_path, args.output_path, args.region)
    

if __name__ == "__main__":
    main()
