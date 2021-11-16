from datetime import datetime

from sqlalchemy import desc, asc

from src.database.DbManager import cutSensors, up_sensors_tids, sensorBoundGrid
from src.database.Models import sensorMerge, populateTables
from src.map.Central import *
from src.map.HexGrid import geo_tiles_from_db, genSensorLayer_db, geojson_from_tiles


def unique_tiles(tiles):
    unique_tiles = {}

    for t in tiles:
        if t.tid not in unique_tiles.keys():
            unique_tiles[t.tid] = t

    return [unique_tiles[k] for k, v in unique_tiles.items()]


def tile_ranges(sensor_tiles, r=5):
    """
    given list of tiles and range, returns a list of unique tiles that are in range of each
    tile provided.
    """
    tiles = []
    for t in sensor_tiles:
        neighbors = t.tiles_in_range(r)  # this includes the center tile: t
        tiles.extend(neighbors)
        print(f"\tgot {len(neighbors)} neighbors for tile {t.tid}")

    return unique_tiles(tiles)


def main():
    # c = Central('..\\..\\..\\AirBots\\docs\\Model1')
    # data = getMeasuresORM(11571, datetime.datetime(2019, 1, 1, 0), datetime.datetime(2019, 1, 30, 0))
    # for e in data:
    #     print(e)
    # print(len(data))
    # c.sensorSummary()

    tiles = getSensorTiles()
    dist = 3
    clusters = tile_ranges(tiles, r=dist)
    print(len(clusters))
    geojson_from_tiles(clusters, f"sensorTiles_{dist}_ranges.geojson")
    # geojson_from_tiles(circle, f"{id}_{r}_range.geojson")
    # genSensorLayer_db()
    # print("finished making sensor geojson")
    #
    # geo_tiles_from_db()
    # print("finished making tiles geojson")





if __name__ == '__main__':
    main()
