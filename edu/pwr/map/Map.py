class Map:

    # collection of tiles-> collection tiles with coords and elevation.
    tileMesh = []
    aggregationOptions = []

    def __init__(self, map_ID, name, coord_NW, coord_NE, coord_SW, coord_SE, ):
        self.map_ID = map_ID
        self.coord_NW = coord_NW
        self.coord_NE = coord_NE
        self.coord_SW = coord_SW
        self.coord_SE = coord_SE
        self.name = name


# method for generating tile mesh
def createMesh():
    pass


# getter for all fields in tile
def getTileInfo(tile_id):
    pass


# print aggregations
def displayAggeregations(*args):
    pass


def classifyTiles():
    pass