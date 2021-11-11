import geopy as gp

from src.database.Models import Tile
from src.map.MapPoint import MapPoint
from src.map.TileBin import calcCoordinate
import math

cf = {'green': 0.50,
      'water': 0.75,
      'residential': 1.10,
      'industrial': 1.88,
      'commercial': 1.25}


def getCF(tile: Tile):
    return cf[tile.tclass]


# hex w = sqrt(3) * size (distance from H1 center to H2 center)
# hex h = 2 * size
# size = 50 (center to vertex, aka radius)
# long diameter = 100
def tilesBetween(tile_start: Tile, tile_end: Tile):
    tiles = [tile_start,]
    cfs = [getCF(tile_start),]
    tile_r = 50
    tile_w = math.sqrt(3) * tile_r
    start_mp = MapPoint(latitude=float(tile_start.center_lat), longitude=float(tile_start.center_lon))
    end_coors = tile_end.center.split(",")
    end_mp = MapPoint(latitude=float(end_coors[0]), longitude=float(end_coors[1]))
    angle = start_mp.bearing(end_mp)

    tile_next = tile_start
    done = False
    while not done:
        tile_coors = str(tile_next.center).split(",")
        center_mp = MapPoint(latitude=float(tile_coors[0]), longitude=float(tile_coors[1]))
        next_center = calcCoordinate(center_mp, degrees=angle, dist=tile_w)
        tile_next = fetchTile(next_center)
        tiles.append(tile_next)
        cfs.append(getCF(tile_next))

        if tile_next == tile_end:
            done = True

    return tiles, cfs


def fetchTile(center: MapPoint):
    from src.database.DbManager import Session
    from sqlalchemy import and_

    bounds = bounding_box(center)
    with Session as sesh:
        tile = sesh.query(Tile).filter(and_(Tile.center_lat >= bounds['s'], Tile.center_lat <= bounds['n'])). \
                                filter(and_(Tile.center_lon >= bounds['w'], Tile.center_lin <= bounds['e'])). \
                                all()

    return tile


def bounding_box(mp: MapPoint, radius=20):
    # S-, N+
    lats = []
    # W-, E+
    lons = []
    # degrees
    for x in range(0, 360, 360//4):
        new_mp = calcCoordinate(mp, radius, x)
        lats.append(new_mp.latitude)
        lons.append(new_mp.longitude)

    N = max(lats)
    S = min(lats)
    E = max(lons)
    W = min(lons)

    # bb = BoundBox(N, S, E, W)

    return {"n": N, "s": S, "e": E, "w": W}


class BoundBox:
    def __init__(self, N, S, E, W):
        self.ne = MapPoint(latitude=N, longitude=E)
        self.se = MapPoint(latitude=S, longitude=E)
        self.sw = MapPoint(latitude=S, longitude=W)
        self.nw = MapPoint(latitude=N, longitude=W)

    def contains(self, mp:MapPoint):
        from shapely.geometry import Polygon, Point
        pt = Point(mp.lon, mp.lat)
        vertices = [
                (self.ne.lon, self.ne.lat),
                (self.se.lon, self.se.lat),
                (self.sw.lon, self.sw.lat),
                (self.nw.lon, self.nw.lat)
        ]
        bb = Polygon(vertices)

        return bb.contains(pt)


if __name__ == '__main__':
    bounding_box()

