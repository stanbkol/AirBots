import json
from urllib.parse import urlencode
from math import sqrt
import geojson
import requests
from flask import Flask
import folium
from geojson import Polygon, Feature, FeatureCollection, Point
from src.map.MapPoint import calcDistance, MapPoint, calcCoordinate
from src.map.TileBin import TileBin
from src.database.utils import drange
from src.database.Models import getTilesORM
from config import config

# pip install wheel
# pip install pipwin
#
# pipwin install gdal
# pipwin install fiona
# pipwin install geopandas
# pip install kelpler
import keplergl

app = Flask(__name__)


@app.route("/")
def base():
    opole_map = folium.Map(location=[50.67211, 17.92533],
                           zoom_start=13, height=800
                           )
    geoJson_op = folium.GeoJson('..\\..\\AirBots\\geojsons\\tiles_layer.geojson',
                                name='opole tiles')
    geoJson_op.add_to(opole_map)

    folium.GeoJson('..\\..\\AirBots\\geojsons\\city_layer.geojson', name='opole box').add_to(opole_map)

    folium.GeoJson('..\\..\\AirBots\\geojsons\\sensor_layer.geojson', name='sensors').add_to(opole_map)

    # add layer control to map (allows layer to be turned on or off)
    folium.LayerControl().add_to(opole_map)

    return opole_map._repr_html_()


@app.route("/kepler")
def kepler():
    opole = setKeplerMap()

    return opole._repr_html_()

# TODO: swtich to ORM querying
def getSensorCoords():
    # lon lat order
    pass


def setKeplerMap():
    with open('..\\..\\AirBots\\geojsons\\sensor_layer.geojson', 'r') as f:
        sensors = f.read()

    with open('..\\..\\AirBots\\geojsons\\tiles_layer.geojson', 'r') as f:
        tiles = f.read()

    with open('..\\..\\AirBots\\geojsons\\filledTiles_layer.geojson', 'r') as f:
        tiles_filled = f.read()

    opole = keplergl.KeplerGl(height=600, config=config)
    if sensors:
        opole.add_data(sensors, "sensors")
    if tiles:
        opole.add_data(tiles, "tiles")
    if tiles_filled:
        opole.add_data(tiles_filled, "tiles with sensors")

    print(opole.config)

    return opole

def genCityLayer():
    map_nw = (50.76997429, 17.77959063)
    map_ne = (50.76997429, 18.03269049)
    map_se = (50.58761735, 18.03269049)
    map_sw = (50.58761735, 17.77959063)
    opole_bounds = [map_nw, map_ne, map_se, map_sw, map_nw]

    # Long Lat order
    opole_bounds = [(t[1], t[0]) for t in opole_bounds]

    boxed_city = Polygon([opole_bounds])
    map_polys = [boxed_city]

    map_features = []

    for p in map_polys:
        map_features.append(Feature(geometry=p))

    map_collection = FeatureCollection(map_features)

    with open('..\\..\\AirBots\\geojsons\\city_layer.geojson', "w") as out:
        geojson.dump(map_collection, out)


def getPolys(tiles, lonlat=False):
    polys = []
    for t in tiles:
        coords = t.getVertices()
        coords.append(coords[0])
        # if true, swaps order of coordinates to longitude, latitude
        if lonlat:
            coords = [(c[1], c[0]) for c in coords]

        polys.append(Polygon(coords))

    return polys


def geo_tiles_from_db():
    tiles = getTilesORM()
    tile_polys = getPolys(tiles, lonlat=True)
    tile_features = []
    for poly in tile_polys:
        f = Feature(geometry=poly)
        tile_features.append(f)
    print("saving..")
    tile_collection = FeatureCollection(tile_features)
    # C:\\Users\\stanb\\PycharmProjects
    with open('..\\..\\AirBots\\geojsons\\tiles_layer.geojson', "w") as out:
        geojson.dump(tile_collection, out)


def genSensorLayer():
    sensor_feats = []
    sensorsLonLat = getSensorCoords()
    print(sensorsLonLat)
    sensor_markers = []
    for s in sensorsLonLat:
        pt = Point(s)
        sensor_markers.append(pt)

    for p in sensor_markers:
        f = Feature(geometry=p)
        sensor_feats.append(f)

    sensor_featCollection = FeatureCollection(sensor_feats)

    with open('..\\..\\AirBots\\geojsons\\sensor_layer.geojson', "w") as out:
        geojson.dump(sensor_featCollection, out)


def create_layers():
    pass


def genHexGrid():
    hex = 6
    tile_d = 100
    tile_r = tile_d / 2
    hex_h_coef = 3 / 2
    hex_w_coef = sqrt(3)

    map_nw = (50.76997429, 17.77959063)
    map_ne = (50.76997429, 18.03269049)
    map_se = (50.58761735, 18.03269049)
    map_sw = (50.58761735, 17.77959063)

    # Long Lat order
    width_m = calcDistance(map_nw, map_ne)
    height_m = calcDistance(map_nw, map_sw)

    y_jump = hex_h_coef * tile_d
    x_jump = tile_r * hex_w_coef
    startLL = MapPoint(map_nw[0], map_nw[1])

    tiles = []
    tid = 0
    # first row -- not indented
    print("creating tiles..")
    for y in drange(0, height_m, y_jump):
        for d in drange(0, width_m, x_jump):
            center = calcCoordinate(calcCoordinate(startLL, y, 180), d, 90)
            tid += 1
            tile = TileBin(tileID=int(tid), mapID=1, center=center, numSides=hex, diameter=tile_d)
            mapPoints = tile.generate_vertices_coordinates()
            vertices = [mp.LonLatCoords for mp in mapPoints]

            # add first vertex to close the polygon
            vertices.append(vertices[0])

            # add remaining data to tile for postgresql insert
            poly_str = create_poly_string(vertices)
            tile.set_PolyString(poly_str)
            tiles.append(tile)


    # indented tiles
    startLL = calcCoordinate(calcCoordinate(startLL, 3 / 4 * tile_d, 180), x_jump / 2, 90)
    for y in drange(0, height_m - y_jump, y_jump):
        for d in drange(0, width_m, x_jump):
            center = calcCoordinate(calcCoordinate(startLL, y, 180), d, 90)
            tid += 1
            tile = TileBin(mapID=1, tileID=int(tid), center=center, numSides=hex, diameter=tile_d)
            mapPoints = tile.generate_vertices_coordinates()
            vertices = [mp.LonLatCoords for mp in mapPoints]

            vertices.append(vertices[0])

            # add remaining data to tile for postgresql insert
            poly_str = create_poly_string(vertices)
            tile.set_PolyString(poly_str)
            tiles.append(tile)

    print("# tiles: " + str(len(tiles)))
    print("inserting tiles to table..")
    # conn = createConnection()

    # for tile in tiles:
        # insertTile(conn, tile)

    # TODO: update to insert into new ORM tables


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


if __name__ == "__main__":
    # genHexGrid()
    # geo_tiles_from_db(1)
    # genSensorLayer()
    # create_layers()
    app.run(debug=True)


