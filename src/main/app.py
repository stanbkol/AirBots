from datetime import datetime

from sqlalchemy import desc

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

    # with Session as sesh:
    #     item = sesh.query(Tile.x).where(Tile.y % 2 == 1).order_by(desc(Tile.x)).first()[0]
    # print(item)
    start = getTileCellORM(283, 1)
    end = getTileCellORM(283, 5)
    print(str(start), str(end))
    path = start.pathTo(end)
    geojson_from_tiles(path)



if __name__ == '__main__':
    main()
