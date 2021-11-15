from datetime import datetime

from sqlalchemy import desc, asc

from src.database.DbManager import cutSensors, up_sensors_tids
from src.database.Models import sensorMerge, populateTables
from src.map.Central import *
from src.map.HexGrid import geo_tiles_from_db, genSensorLayer, geojson_from_tiles


def main():
    # c = Central('..\\..\\..\\AirBots\\docs\\Model1')
    # data = getMeasuresORM(11571, datetime.datetime(2019, 1, 1, 0), datetime.datetime(2019, 1, 30, 0))
    # for e in data:
    #     print(e)
    # print(len(data))
    # c.sensorSummary()

    # up_sensors_tids()
    # cutSensors()

    start = getTileCellORM(10, 10)
    n=3
    bigHex = start.tiles_in_range(n)
    # end = getTileCellORM(283, 5)
    # print(str(start), str(end))
    # path = start.pathTo(end)
    fn = f'range_{n}_of_{start.tid}.geojson'
    geojson_from_tiles(bigHex, fn=fn)



if __name__ == '__main__':
    main()
