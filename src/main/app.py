from datetime import datetime

from sqlalchemy import desc, asc
from src.map.TileClassifier import *
from src.map.Central import *
from src.database.Models import *
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


def checkTiles():
    data = getTilesORM()
    tile_class = {}
    for d in data:
        print(d.tclass)
        if d.tclass not in tile_class.keys():
            tile_class[d.tclass] = 1
        else:
            tile_class[d.tclass] += 1
    print(tile_class)


def classifyTiles():
    data = getJson(r'C:\Users\mrusieck\PycharmProjects\AirBot\docs\tile_scrape.txt')
    for k in data:
        print("Tile #", k)
        l1, l2 = classifyT(data[k])
        updateTileClass(k, l1, l2)


def main():
    # c = Central('..\\..\\..\\AirBots\\docs\\Model1')
    # classifyTiles()
    # checkTiles()
    # target = getTileCellORM(202, 92)

    ra = 3
    agent_tiles_nn = tile_ranges(getSensorTiles(), r=ra)

    fn = f"agentTiles_NR_{ra}.geojson"
    geojson_from_tiles(agent_tiles_nn, fn)

    no_class = []


if __name__ == '__main__':
    main()
