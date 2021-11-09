import geopy as gp
from src.database.Models import Tile
from src.map.MapPoint import MapPoint

def tilesBetween(tile_start: Tile, tile_end: Tile):
    tiles = []
    change_factors = []

    start_coors = str(tile_start.center).split(",")
    start_mp = MapPoint(latitude=float(start_coors[0]), longitude=float(start_coors[1]))
    end_coors = tile_end.center.split(",")
    end_mp = MapPoint(latitude=float(end_coors[0]), longitude=float(end_coors[1]))

    angle = find_bearing(start_mp, end_mp)


