import json
import math
from urllib.parse import urlencode

import geojson
import requests
from geojson import Polygon, Feature, FeatureCollection, Point
from src.main.utils import drange
from src.map.MapPoint import calcCoordinate, calcDistance, MapPoint


class DwHex:
    """
    used in pathfinding between tiles. represents double-width coordinates (x+y)%2==0
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f'<Hex>(x=%s, y=%s)' % (self.x, self.y)


class Hex:
    """
    ysed in pathfinding between tiles. represents cube coordinates q,r,s
    """
    def __init__(self, q, r, s):
        self.q = q
        self.r = r
        self.s = s

    def __str__(self):
        return f'<Hex>(q=%s, r=%s, s=%s)' % (self.q, self.r, self.s)

    def cube_subtract(self, b):
        return Hex(self.q - b.q, self.r - b.r, self.s - b.s)

    def add_cube(self, b):
        return Hex(self.q + b.q, self.r + b.r, self.s + b.s)


def hex_add(a, b):
    return Hex(a.q + b.q, a.r + b.r, a.s + b.s)


def hex_subtract(a, b):
    return Hex(a.q - b.q, a.r - b.r, a.s - b.s)


def hex_scale(a, k):
    return Hex(a.q * k, a.r * k, a.s * k)


def hex_rotate_left(a):
    return Hex(-a.s, -a.q, -a.r)


def hex_rotate_right(a):
    return Hex(-a.r, -a.s, -a.q)


hex_directions = [Hex(1, 0, -1), Hex(1, -1, 0), Hex(0, -1, 1), Hex(-1, 0, 1), Hex(-1, 1, 0), Hex(0, 1, -1)]


def hex_direction(direction):
    return hex_directions[direction]


def hex_neighbor(hex, direction):
    return hex_add(hex, hex_direction(direction))


hex_diagonals = [Hex(2, -1, -1), Hex(1, -2, 1), Hex(-1, -1, 2), Hex(-2, 1, 1), Hex(-1, 2, -1), Hex(1, 1, -2)]


def hex_diagonal_neighbor(hex, direction):
    return hex_add(hex, hex_diagonals[direction])


def hex_length(hex):
    return (abs(hex.q) + abs(hex.r) + abs(hex.s)) // 2


def hex_distance(a, b):
    return hex_length(hex_subtract(a, b))


def hex_round(h):
    qi = int(round(h.q))
    ri = int(round(h.r))
    si = int(round(h.s))
    q_diff = abs(qi - h.q)
    r_diff = abs(ri - h.r)
    s_diff = abs(si - h.s)
    if q_diff > r_diff and q_diff > s_diff:
        qi = -ri - si
    else:
        if r_diff > s_diff:
            ri = -qi - si
        else:
            si = -qi - ri
    return Hex(qi, ri, si)


def hex_lerp(a, b, t):
    return Hex(a.q * (1.0 - t) + b.q * t, a.r * (1.0 - t) + b.r * t, a.s * (1.0 - t) + b.s * t)


def right_nudge(a: Hex):
    return Hex(a.q + 1e-06, a.r + 1e-06, a.s - 2e-06)


def left_nudge(b: Hex):
    return Hex(b.q - 1e-06, b.r - 1e-06, b.s + 2e-06)


def hex_linedraw(a, b, right_edge=False):
    """
    finds and lists a direct Tile path between two Tiles.
    :param a: starting tile
    :param b: ending tile
    :param right_edge: if start and end are on the far right of edge of map, set to True.
    :return: list of tiles, start and end tiles included
    """
    N = int(hex_distance(a, b))
    # if on the right edge flip +/-
    if right_edge:
        a_nudge = left_nudge(a)
        b_nudge = left_nudge(b)
    else:
        a_nudge = right_nudge(a)
        b_nudge = right_nudge(b)

    results = []
    step = 1.0 / max(N, 1)
    for i in range(0, N + 1):
        results.append(hex_round(hex_lerp(a_nudge, b_nudge, step * i)))
    return results


def neighbors_in_range(center: Hex, N):
    results = []
    for q in range(-N, N+1, 1):
        for r in range(max(-N, -q - N), min(+N, -q+N)+1, 1):
            s = -q - r
            results.append(hex_add(center, Hex(q, r, s)))

    return results

def dw_to_hex(dw):
    q = (dw.x - dw.y) / 2
    r = dw.y
    s = -1 * q - r
    return Hex(q, r, s)


def hex_to_dw(h):
    y = 2 * h.q + h.r
    x = h.r
    return DwHex(y, x)


def dw_distance(a, b):
    dy = abs(a.y - b.y)
    dx = abs(a.x - b.x)
    return dy + max(0, (dx-dy)/2)


def genCityLayer(bounds):
    """
    creates a single rectangle representing the bounding box of a city. saves to geojson file.
    :param bounds: the n,s,e,w geo-coordinate bounds in latitude and longitude decimal notation.
    :return:
    """
    # lat, lon
    ne = (bounds['n'], bounds['e'])
    se = (bounds['s'], bounds['e'])
    sw = (bounds['s'], bounds['w'])
    nw = (bounds['n'], bounds['w'])

    # map_nw = (50.76997429, 17.77959063)
    # map_ne = (50.76997429, 18.03269049)
    # map_se = (50.58761735, 18.03269049)
    # map_sw = (50.58761735, 17.77959063)
    city_bounds = [nw, ne, se, sw, nw]

    # Lon Lat order
    city_bounds = [(t[1], t[0]) for t in city_bounds]

    boxed_city = Polygon([city_bounds])
    map_polys = [boxed_city]

    map_features = []

    for p in map_polys:
        map_features.append(Feature(geometry=p))

    map_collection = FeatureCollection(map_features)

    with open('..\\..\\AirBots\\geojsons\\city_layer.geojson', "w") as out:
        geojson.dump(map_collection, out)


def getPolys(tiles, lonlat=False):
    """
    create a list of geojson polygons, provided Tile models.
    :param tiles: list of Tile models
    :param lonlat: sets the order of Polygon coordinates. default is latitude, longitude
    :return: list of geojson polygons.
    """
    polys = []
    for t in tiles:
        coords = t.getVertices()
        coords.append(coords[0])
        # if true, swaps order of coordinates to longitude, latitude
        if lonlat:
            coords = [(c[1], c[0]) for c in coords]

        polys.append(Polygon([coords]))

    return polys


def geo_tiles_from_db():
    """
    fetches Tile models from the database and creates a geojson of hexagons
    :return:
    """
    from src.database.Models import getTilesORM
    tiles = getTilesORM()
    print(f'num tiles: %s' % len(tiles))
    # tile_polys = getPolys(tiles, lonlat=False)
    tile_features = []
    for t in tiles:
        vertices = t.getVertices(lonlat=True)
        vertices.append(vertices[0])
        poly = Polygon([vertices])
        tile_class = t.tclass if t.tclass else "N/A"
        properties = {"id": t.tid,
                      "lat_lon": f'({t.center_lat}, {t.center_lon})',
                      "grid_xy": f'({t.x}, {t.y})',
                      "class": tile_class
                      }
        f = Feature(geometry=poly, properties=properties)
        tile_features.append(f)
    print("saving..")
    tile_collection = FeatureCollection(tile_features)
    with open('..\\..\\..\\AirBots\\geojsons\\tiles_layer_props.geojson', "w") as out:
        geojson.dump(tile_collection, out)


def genSensorLayer_db():
    """
    create a geojson for Sensor point markers
    :return:
    """
    from src.database.Models import getSensorsORM
    sensor_feats = []
    sensorsLonLat = [(s.lon, s.lat) for s in getSensorsORM()]
    # print(len(sensorsLonLat))
    sensor_markers = [Point(s) for s in sensorsLonLat]
    for sensor in getSensorsORM():
        coords = (sensor.lon, sensor.lat)
        tile_id = sensor.tid if sensor.tid else "None"
        marker = Point(coords)
        properties = {"id:": sensor.sid,
                      "lat_lon": f'({sensor.lat},{sensor.lon})',
                      "tile_id": tile_id
                    }
        f = Feature(geometry=marker, properties=properties)
        sensor_feats.append(f)

    # for p in sensor_markers:
    #     f = Feature(geometry=p)
    #     sensor_feats.append(f)

    sensor_featCollection = FeatureCollection(sensor_feats)

    with open('..\\..\\..\\AirBots\\geojsons\\sensor_layer_props.geojson', "w") as out:
        geojson.dump(sensor_featCollection, out)


def create_layers():
    return NotImplemented()


def geojson_from_tiles(tiles, fn=None):
    if not fn:
        start_sid = tiles[0].tid
        end_sid = tiles[-1].tid
        fn = f'path_%s_to_%s.geojson' % (start_sid, end_sid)

    tile_features = []
    for t in tiles:
        vertices = t.getVertices(lonlat=True)
        vertices.append(vertices[0])
        poly = Polygon([vertices])
        tile_class = t.tclass if t.tclass else "N/A"
        properties = {"id": t.tid,
                      "coords": f'({t.center_lat}, {t.center_lon})',
                      "grid_xy": f'({t.x}, {t.y})',
                      "class": tile_class
                      }
        f = Feature(geometry=poly, properties=properties)
        tile_features.append(f)
    print("saving..")
    tile_collection = FeatureCollection(tile_features)
    with open('..\\..\\..\\AirBots\\geojsons\\'+fn, "w") as out:
        geojson.dump(tile_collection, out)


def genHexGrid(bounds):
    """
    generates a grid of (100m long diagonal) pointed hexagons for the provided geo-coordinate bounds.
    hexagons are mapped using the double-width coordinate system.
    :param bounds: dict of n,s,e,w geo-coordinate bounds of map
    :return: list of Tile objects
    """
    from src.database.Models import Tile
    hex = 6
    tile_d = 100
    tile_r = tile_d / 2
    hex_h_coef = 3 / 2
    hex_w_coef = math.sqrt(3)

    # Opole bb
    # map_nw = (50.76997429, 17.77959063)
    # map_ne = (50.76997429, 18.03269049)
    # map_se = (50.58761735, 18.03269049)
    # map_sw = (50.58761735, 17.77959063)

    # ne = MapPoint(latitude=bounds['n'], longitude=bounds['e'])
    # se = MapPoint(latitude=bounds['s'], longitude=bounds['e'])
    # sw = MapPoint(latitude=bounds['s'], longitude=bounds['w'])
    # nw = MapPoint(latitude=bounds['n'], longitude=bounds['w'])

    ne = (bounds['n'], bounds['e'])
    se = (bounds['s'], bounds['e'])
    sw = (bounds['s'], bounds['w'])
    nw = (bounds['n'], bounds['w'])

    # Long Lat order
    width_m = calcDistance(nw, ne)
    print(f'width: %s' % width_m)

    height_m = calcDistance(nw, sw)
    print(f'height: %s' % height_m)

    y_jump = hex_h_coef * tile_d
    x_jump = tile_r * hex_w_coef
    startLL = MapPoint(latitude=nw[0], longitude=nw[1])

    tiles = []
    tid = 0
    # first row -- not indented
    y_coord = 0
    print("creating tiles..")
    for y in drange(0, height_m, y_jump):
        x_coord = 0
        for d in drange(0, width_m, x_jump):
            center = calcCoordinate(calcCoordinate(startLL, y, 180), d, 90)
            tid += 1
            tile = Tile(tileID=int(tid), mapID=1, center_lat=center.lat, center_lon=center.lon, numSides=hex,
                        diameter=tile_d, xaxis=x_coord, yaxis=y_coord)
            mapPoints = tile.generate_vertices_coordinates()

            # add remaining data to tile for postgresql insert
            tiles.append(tile)
            x_coord += 2

        y_coord += 2

    # indented tiles
    y_coord = 1
    startLL = calcCoordinate(calcCoordinate(startLL, 3 / 4 * tile_d, 180), x_jump / 2, 90)
    for y in drange(0, height_m - y_jump, y_jump):
        x_coord = 1
        for d in drange(0, width_m, x_jump):
            center = calcCoordinate(calcCoordinate(startLL, y, 180), d, 90)
            tid += 1
            tile = Tile(tileID=int(tid), mapID=1, center_lat=center.lat, center_lon=center.lon, numSides=hex,
                        diameter=tile_d, xaxis=x_coord, yaxis=y_coord)
            mapPoints = tile.generate_vertices_coordinates()

            # add remaining data to tile for postgresql insert
            tiles.append(tile)
            x_coord += 2

        y_coord += 2

    print("# tiles: " + str(len(tiles)))
    print("done generating Tiles")

    return tiles


def getGeocoding(sensor):
    address = sensor.address1 + ' Opole, Poland'
    AUTH_KEY = 'AIzaSyDpiyrbHH1OE5f0YKvEh2xvTrhcPBUzuCI'
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
    params = {'address': address,
              'key': AUTH_KEY}
    # f"{base_url}{urllib.parse.urlencode(params)}
    r = requests.get(base_url, params=params)
    data = json.loads(r.content)
    return data


def extract_lat_lng(sensor, data_type='json'):
    api_key = 'AIzaSyDpiyrbHH1OE5f0YKvEh2xvTrhcPBUzuCI'

    loc_query = sensor.address1 + ', Opole, Poland'

    endpoint = f"https://maps.googleapis.com/maps/api/geocode/{data_type}"
    params = {"address": loc_query, "key": api_key}
    url_params = urlencode(params)
    url = f"{endpoint}?{url_params}"
    r = requests.get(url)
    if r.status_code not in range(200, 299):
        return {}
    latlng = {}
    try:
        latlng = r.json()['results'][0]['geometry']['location']
    except:
        pass
    lat, lng = latlng.get("lat"), latlng.get("lng")
    return lat, lng


def create_marker_string(long, lat):
    return "Point({p})".format(p=[(long, lat)])


def create_poly_string(longlat_list):
    return "Polygon({coord_list})".format(coord_list=[longlat_list])


def tile_dist_trust_factors(max_range):
    step = 0.98 / max(max_range, 1)
    dist_factors = [round(s * step, 3) for s in range(1, max_range+1)]
    dist_factors.reverse()
    return dist_factors

