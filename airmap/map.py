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
from src.database.Models import getTilesORM, Tile

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


# TODO: swtich to ORM querying
def getSensorCoords():
    # lon lat order
    pass





if __name__ == "__main__":
    # genHexGrid()
    # geo_tiles_from_db(1)
    # genSensorLayer()
    # create_layers()
    app.run(debug=True)


