from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import relationship

from src.database.DbManager import Base


class Map(Base):
    __tablename__ = 'maps'
    __table_args__ = {"schema": "agents"}

    # collection of tiles-> collection tiles with coords and elevation.
    tileMesh = []
    aggregationOptions = []

    map_ID = Column('map_id', Integer,  primary_key=True)
    name = Column('name', String(20))
    coord_NW = Column('Coordinates_NW', String(50))
    coord_NE = Column('Coordinates_NE', String(50))
    coord_SW = Column('Coordinates_SW', String(50))
    coord_SE = Column('Coordinates_SE', String(50))
    tiles = relationship('Tile', backref='Map', lazy='dynamic')

    def __init__(self, map_ID, name, coord_NW, coord_NE, coord_SW, coord_SE, ):
        self.map_ID = map_ID
        self.name = name
        self.coord_NW = coord_NW
        self.coord_NE = coord_NE
        self.coord_SW = coord_SW
        self.coord_SE = coord_SE
        


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