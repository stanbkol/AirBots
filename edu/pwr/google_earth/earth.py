import pydeck as pdk
import ee
import folium

def add_ee_layer(self, ee_image_object, vis_params, name):
    """Adds a method for displaying Earth Engine image tiles to folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)


def run():
    # Trigger the authentication flow.
    # ee.Authenticate()

    # Initialize the library.
    ee.Initialize()
    # Import the MODIS land cover collection.
    lc = ee.ImageCollection('MODIS/006/MCD12Q1')

    # Import the MODIS land surface temperature collection.
    lst = ee.ImageCollection('MODIS/006/MOD11A1')

    # Import the USGS ground elevation image.
    elv = ee.Image('USGS/SRTMGL1_003')

    # Initial date of interest (inclusive).
    i_date = '2017-01-01'

    # Final date of interest (exclusive).
    f_date = '2020-01-01'

    # Selection of appropriate bands and dates for LST.
    lst = lst.select('LST_Day_1km', 'QC_Day').filterDate(i_date, f_date)

    # folium.Map.add_ee_layer = add_ee_layer

    # Select a specific band and dates for land cover.
    lc_img = lc.select('LC_Type1').filterDate(i_date).first()

    # Set visualization parameters for land cover.
    lc_vis_params = {
        'min': 1, 'max': 17,
        'palette': ['05450a', '086a10', '54a708', '78d203', '009900', 'c6b044',
                    'dcd159', 'dade48', 'fbff13', 'b6ff05', '27ff87', 'c24f44',
                    'a5a5a5', 'ff6d4c', '69fff8', 'f9ffa4', '1c0dff']
    }

    # Create a map.
    lat, lon = 45.77, 4.855
    my_map = folium.Map(location=[lat, lon], zoom_start=7)
    my_map.add_ee_layer = add_ee_layer
    # Add the land cover to the map object.
    my_map.add_ee_layer(lc_img, lc_vis_params, 'Land Cover')

    # Add a layer control panel to the map.
    my_map.add_child(folium.LayerControl())

    # Display the map.
    my_map


def simpleMap():
    ee.Initialize()

    opole = [50.67211, 17.92533]

    opole_map = folium.Map(location=opole, zoom_start=4)





if __name__ == '__main__':
    run()