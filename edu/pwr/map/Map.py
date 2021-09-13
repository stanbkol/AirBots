from sqlalchemy import create_engine, Column, String, Integer, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

base = declarative_base()


class Map:
    __tablename__ = 'maps'
    __table_args__ = {"schema": "airbots"}

    # collection of tiles-> collection tiles with coords and elevation.
    tileMesh = []
    aggregationOptions = []

    map_ID = Column('map_id', Integer,  primary_key=True)
    name = Column('name', String(20))
    coord_NW = Column('C_NW', Float)
    coord_NE = Column('C_NE', Float)
    coord_SW = Column('C_SW', Float)
    coord_SE = Column('C_SE', Float)

    def __init__(self, coord_NW, coord_NE, coord_SW, coord_SE, map_ID, name):
        self.coord_NW = coord_NW
        self.coord_NE = coord_NE
        self.coord_SW = coord_SW
        self.coord_SE = coord_SE
        self.map_ID = map_ID
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