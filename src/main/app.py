from datetime import datetime

from sqlalchemy import desc, asc

from src.database.DbManager import cutSensors, up_sensors_tids, sensorBoundGrid
from src.database.Models import sensorMerge, populateTables
from src.map.Central import *
from src.map.HexGrid import geo_tiles_from_db, genSensorLayer_db, geojson_from_tiles


def main():
    # c = Central('..\\..\\..\\AirBots\\docs\\Model1')
    # data = getMeasuresORM(11571, datetime.datetime(2019, 1, 1, 0), datetime.datetime(2019, 1, 30, 0))
    # for e in data:
    #     print(e)
    # print(len(data))
    # c.sensorSummary()
    # up_sensors_tids()
    # sensorBoundGrid()

    id  = 20947
    tile = getTileORM(id)
    r = 3
    circle = tile.tiles_in_range(r)
    geojson_from_tiles(circle, f"{id}_{r}_range.geojson")
    # genSensorLayer_db()
    # print("finished making sensor geojson")
    #
    # geo_tiles_from_db()
    # print("finished making tiles geojson")


if __name__ == '__main__':
    main()
