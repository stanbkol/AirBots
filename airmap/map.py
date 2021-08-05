from urllib.parse import urlencode, urlparse, parse_qsl
from math import sqrt
import geojson
import requests
from flask import Flask
import folium
from geojson import Polygon, Feature, FeatureCollection, Point
from edu.pwr.database.DataLoader import createConnection, fetchValidSensors
from edu.pwr.map.MapPoint import calcDistance, MapPoint, calcCoordinate
from edu.pwr.map.TileBin import drange, TileBin

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

def getSensorCoords():
    conn = createConnection()
    sensorsData = fetchValidSensors(conn)
    print(sensorsData[0])
    sensorCoords = [(t[6], t[5]) for t in sensorsData]
    return sensorCoords


def genHexGeoJson():
    hex = 6
    tile_d = 500
    tile_r = tile_d / 2
    polys = []

    conn = createConnection()
    map_nw = (50.76997429, 17.77959063)
    map_ne = (50.76997429, 18.03269049)
    map_se = (50.58761735, 18.03269049)
    map_sw = (50.58761735, 17.77959063)
    opole_bounds = [map_nw, map_ne, map_se, map_sw, map_nw]

    # Long Lat order
    opole_bounds = [(t[1], t[0]) for t in opole_bounds]

    boxed_city = Polygon([opole_bounds])
    map_polys = [boxed_city]
    # polys.append(boxed_city)

    width_m = calcDistance(map_nw, map_ne)
    height_m = calcDistance(map_nw, map_sw)

    border_tiles = []
    offset_tiles = []

    y_jump = 3/2 * tile_d
    x_jump = tile_r * sqrt(3)
    startLL = MapPoint(map_nw[0], map_nw[1])
    # center = calcCoordinate(calcCoordinate(startLL, 0, 180), 0, 90)
    #
    # tile = TileBin(1, 1, center, hex)
    # mapPoints = tile.generate_vertices_coordinates()
    # vertices = [mp.LonLatCoords for mp in mapPoints]
    # vertices.append(vertices[0])
    #
    # polys.append(Point(center.LonLatCoords))
    # p = Polygon([vertices])
    # polys.append(p)

    for y in drange(0, height_m, y_jump):
        for d in drange(0, width_m, x_jump):
            center = calcCoordinate(calcCoordinate(startLL, y, 180), d, 90)
            tid = y / y_jump * 10 + d / x_jump
            tile = TileBin(1, tid, center, hex, diameter=tile_d)
            mapPoints = tile.generate_vertices_coordinates()
            vertices = [mp.LonLatCoords for mp in mapPoints]

            vertices.append(vertices[0])
            # create Polygon and add it to list
            p = Polygon([vertices])
            polys.append(p)

    startLL = calcCoordinate(calcCoordinate(startLL, 3/4 * tile_d, 180), x_jump/2, 90)
    for y in drange(0, height_m-y_jump, y_jump):
        for d in drange(0, width_m, x_jump):
            center = calcCoordinate(calcCoordinate(startLL, y, 180), d, 90)
            tid = y / y_jump * 10 + d / x_jump
            tile = TileBin(1, tid, center, hex, diameter=tile_d)
            mapPoints = tile.generate_vertices_coordinates()
            vertices = [mp.LonLatCoords for mp in mapPoints]

            vertices.append(vertices[0])
            # create Polygon and add it to list
            p = Polygon([vertices])
            polys.append(p)

    featurelist = []
    map_features = []
    for p in polys:
        f = Feature(geometry=p)
        featurelist.append(f)

    for p in map_polys:
        map_features.append(Feature(geometry=p))

    map_collection = FeatureCollection(map_features)
    fcollect = FeatureCollection(featurelist)

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

    # C:\\Users\\stanb\\PycharmProjects
    with open('..\\..\\AirBots\\geojsons\\tiles_layer.geojson', "w") as out:
        geojson.dump(fcollect, out)

    with open('..\\..\\AirBots\\geojsons\\city_layer.geojson', "w") as out:
        geojson.dump(map_collection, out)

    # sensor coords for markers
    # sensordata = pandas.DataFrame(fetchValidSensors(conn))


# def getGeocoding(sensor):
#     address = sensor.address_1 + ' Opole, Poland'
#     AUTH_KEY = 'AIzaSyDpiyrbHH1OE5f0YKvEh2xvTrhcPBUzuCI'
#     base_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
#     params = {'address': address,
#               'key': AUTH_KEY}
#     # f"{base_url}{urllib.parse.urlencode(params)}
#     r = requests.get(base_url, params=params)
#     data = json.loads(r.content)
#     return data


def extract_lat_lng(sensor, data_type='json'):
        api_key = 'AIzaSyDpiyrbHH1OE5f0YKvEh2xvTrhcPBUzuCI'

        loc_query = sensor.address_1 + ', Opole, Poland'

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


if __name__ == "__main__":
    # t1 = [(1,4),(3,5),(6,9)]
    # t2 = [(3, 3), (5, 5), (9, 9)]
    # vs = []
    # vs.append(t1)
    # vs.append(t2)
    # print(vs)
    #getSensorCoords()
    genHexGeoJson()
    app.run(debug=True)
