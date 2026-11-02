import os
import json
import argparse
import numpy as np

import geopandas as gpd
from shapely.geometry import Polygon

from shapely.geometry import (
    LineString,
    MultiLineString,
    mapping
)

# Class definitions
CLASS_MAP = {
    "divider": 0,
    "ped_crossing": 1,
    "boundary": 2
}

def resample_linestring(
        line,
        num_points):

    if line.length == 0:
        return None

    distances = np.linspace(
        0,
        line.length,
        num_points
    )

    points = []

    for distance in distances:
        point = line.interpolate(
            distance
        )

        points.append(
            (
                point.x,
                point.y
            )
        )

    return LineString(points)

# Extract LineStrings from geometry
def extract_lines(geometry):
    if geometry is None:
        return []

    if geometry.geom_type == "LineString":
        return [geometry]

    elif geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)

    return []

# Create GeoJSON FeatureCollection for tile
def process_tile(
        tile,
        layers,
        num_points):

    fname = tile["fname"]

    width = tile["width"]
    height = tile["height"]

    tile_geom = tile.geometry

    # Tile boundaries are stored as closed LineStrings
    # Convert them into polygons for clipping
    if tile_geom.geom_type == "LineString":
        tile_geom = Polygon(
            tile_geom.coords
        )

    features = []

    for class_name, layer in layers.items():
        class_id = CLASS_MAP[class_name]

        # Find geometries intersecting tile
        candidates = layer[
            layer.intersects(tile_geom)
        ]

        for geom in candidates.geometry:
            clipped = geom.intersection(
                tile_geom
            )

            for line in extract_lines(clipped):
                sampled_line = resample_linestring(
                    line,
                    num_points
                )

                if sampled_line is None:
                    continue

                features.append(
                    {
                        "type":
                            "Feature",
                        "properties":
                            {
                                "class":
                                    class_name,
                                "class_id":
                                    class_id
                            },
                        "geometry":
                            mapping(sampled_line)
                    }
                )

    return {
        "type":
            "FeatureCollection",

        "name":
            "MapTR_labels",

        "crs":
            {
                "type":
                    "name",
                "properties":
                    {
                        "name":
                            "urn:ogc:def:crs:EPSG::3857"
                    }
            },

        "properties":
            {
                "image":
                    fname,
                "width":
                    width,
                "height":
                    height,
                "coordinate_system":
                    "EPSG:3857",
                "num_points":
                    num_points
            },

        "features":
            features
    }

def main(args):

    os.makedirs(
        args.output,
        exist_ok=True
    )

    print("Loading datasets...")

    road_edges = gpd.read_file(
        args.road_edges
    )

    lane_markers = gpd.read_file(
        args.lane_markers
    )

    stop_lines = gpd.read_file(
        args.stop_lines
    )

    tiles = gpd.read_file(
        args.tiles
    )

    road_edges = road_edges.to_crs(
        tiles.crs
    )

    lane_markers = lane_markers.to_crs(
        tiles.crs
    )

    stop_lines = stop_lines.to_crs(
        tiles.crs
    )

    layers = {
        "divider":
            lane_markers,
        "ped_crossing":
            stop_lines,
        "boundary":
            road_edges
    }

    print(f"Generating labels for {len(tiles)} tiles")

    for idx, tile in tiles.iterrows():
        label = process_tile(
            tile,
            layers,
            args.num_points
        )

        outfile = os.path.join(
            args.output,
            label["properties"]["image"]
            .replace(".jp2", ".geojson")
            .replace(".png", ".geojson")
        )

        with open(
            outfile,
            "w"
        ) as f:
            json.dump(
                label,
                f,
                indent=4
            )

        if idx % 100 == 0:
            print(
                f"Processed {idx}/{len(tiles)}"
            )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    CITY = 'kitchener-on-2025'
    MONTH = 'apr'
    NUM_POINTS = 20
    # path to the annotations (geojson)
    parser.add_argument(
        "--road_edges",
        default=f"/media/jerry/Data3/datasets_vertical/{CITY}/verified-files/{MONTH}/RoadEdge_Detection/100_percent/RoadEdge_annotations.geojson",
        # required=False
    )

    # path to the annotations (geojson)
    parser.add_argument(
        "--lane_markers",
        default=f"/media/jerry/Data3/datasets_vertical/{CITY}/verified-files/{MONTH}/LaneMarker_All_Detection/100_percent/LaneMarker_annotations.geojson",
        # required=True
    )

    # path to the annotations (geojson)
    parser.add_argument(
        "--stop_lines",
        default=f"/media/jerry/Data3/datasets_vertical/{CITY}/verified-files/{MONTH}/CrossWalkAll_Detection/100_percent/CrossWalkAll_kitchener_linestring_annotations.geojson",
        # required=True
    )

    # path to 0_generated_tiles.geojson
    parser.add_argument(
        "--tiles",
        default=f"/media/jerry/Data3/datasets_vertical/{CITY}/city-images/{MONTH}/tiles/100_percent/1250_150/0_generated_tiles.geojson",
        # required=True
    )

    # e.g. /home/geomate/repos/datasets_vertical/kitchener-on-2025/polylines/apr/
    parser.add_argument(
        "--output",
        default=f"/media/jerry/Data3/datasets_vertical/{CITY}/polylines/polylines_{NUM_POINTS}",
        # required=True
    )

    #  vary this value. default is 20 coordinate pairs
    parser.add_argument(
        "--num_points",
        type=int,
        default=NUM_POINTS
    )

    args = parser.parse_args()

    main(args)