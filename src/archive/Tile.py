from sqlalchemy.orm import relationship

from src.map.MapPoint import calcCoordinate, calcDistance, MapPoint
from sqlalchemy import Column, String, Integer, Float, ForeignKey
from src.database.DbManager import Base

from src.database.utils import drange


class Tile(Base):
    __tablename__ = 'tiles'
    __table_args__ = {"schema": "agents"}

    tid = Column('tile_id', Integer, primary_key=True)
    mid = Column('map_id', Integer, ForeignKey("agents.maps.map_id"), nullable=False)
    sides = Column('num_sides', Integer)
    center = Column('center_latlon', String(50))
    v1 = Column('vertex1', String(50))
    v2 = Column('vertex2', String(50))
    v3 = Column('vertex3', String(50))
    v4 = Column('vertex4', String(50))
    v5 = Column('vertex5', String(50))
    v6 = Column('vertex6', String(50))
    dm = Column('diameter_m', Float)
    tclass = Column('class', String(50))
    max_elev = Column('max_elevation', Float)
    min_elev = Column('min_elevation', Float)
    temp = Column('temperature_c', Float)
    pm10 = Column('pm10_avg', Float)
    pm1 = Column('pm1_avg', Float)
    pm25 = Column('pm25_avg', Float)
    sensors = relationship('Sensor', backref='Tile', lazy='dynamic')

    # maps = relationship("Map")

    def __init__(self, tileID=None, mapID=None, numSides=None, coordinates=None, diameter=None, center=None,
                 tileClass=None, max_elevation=None, min_elevation=None, temperature=None,
                 pm10_avg=None, pm1_avg=None, pm25_avg=None):
        self.tid = tileID
        self.mid = mapID
        self.sides = numSides
        self.v1 = coordinates[0].latlon_str
        self.v2 = coordinates[1].latlon_str
        self.v3 = coordinates[2].latlon_str
        self.v4 = coordinates[3].latlon_str
        self.v5 = coordinates[4].latlon_str
        self.v6 = coordinates[5].latlon_str
        self.dm = diameter
        self.center = center
        self.tclass = tileClass
        self.max_elev = max_elevation
        self.min_elev = min_elevation
        self.temp = temperature
        self.pm10 = pm10_avg
        self.pm1 = pm1_avg
        self.pm25 = pm25_avg

    def __repr__(self):
        return "<Tile(tileid='%s',mapid='%s', center='%s', type='%s')>" % (self.tid, self.mid, self.center,
                                                                             self.tclass)

    def generate_vertices_coordinates(self):
        vertices = []
        radius = self.diameter / 2
        degs = list(drange(0, 360, 360 / self.numSides))
        for d in degs:
            vertex_coor = calcCoordinate(self.centerPt, radius, d)
            # vertices.append(vertex_coor)
            vertices.append(vertex_coor)

        self.v1 = vertices[0]
        self.v2 = vertices[1]
        self.v3 = vertices[2]
        self.v4 = vertices[3]
        self.v5 = vertices[4]
        self.v6 = vertices[5]

        return vertices

    def setClass(self, tile_class):
        self.tclass = tile_class

    def set_vertices(self, vertex_list):
        if len(vertex_list) == self.numSides:
            for i in self.numSides:
                self.coordinates.append(vertex_list[i])

    def metersTo(self, other):
        if isinstance(other, Tile):
            start = MapPoint.createFromStr(self.center)
            end = MapPoint.createFromStr(other.center)
            return calcDistance(startLL=start, endLL=end)