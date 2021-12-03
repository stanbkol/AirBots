from datetime import datetime

from src.map.Central import *
from src.database.Models import *
from src.main.utils import tile_ranges
from src.map.HexGrid import geojson_from_tiles, genSensorLayer_db


def main():
    c = Central('..\\..\\..\\AirBots\\docs\\Model1')
    c.makePrediction(6494, datetime(2019, 1, 7, 0))



if __name__ == '__main__':
    main()
