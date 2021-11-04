from datetime import datetime

from src.database.DataLoader import getMeasures, getSensors, createConnection, getTiles, getSensor
from src.database.DbManager import Base, Session, addOpoleMap, insertTiles, insertSensors, insertMeasures
from src.map.MapPoint import calcCoordinate, calcDistance, MapPoint
from src.database.utils import drange
from sqlalchemy import Column, String, Integer, Float, ForeignKey, DateTime, update
from sqlalchemy.orm import relationship
from sqlalchemy.future import select
import re


class Measure(Base):
    __tablename__ = 'measures'
    __table_args__ = {"schema": "airbots"}

    dk = Column('datekey', Integer, primary_key=True)
    sid = Column('sensorid', Integer, ForeignKey("airbots.sensors.sensor_id"), primary_key=True)
    date = Column('date', DateTime)
    temp = Column('temperature', Float)
    pm1 = Column('pm1', Float)
    pm10 = Column('pm10', Float)
    pm25 = Column('pm25', Float)

    def __init__(self, date_key=None, sensor_id=None, date=None, pm1=None, pm25=None, pm10=None, temperature=None):
        self.dk = date_key
        self.sid = sensor_id
        self.date = date
        self.temp = temperature
        self.pm1 = pm1
        self.pm10 = pm10
        self.pm25 = pm25

    def __str__(self):
        return "Datekey=" + str(self.dk) + " SID=" + str(self.sid) + " temp,pm1,pm10,pm25=(" + str(self.temp) + ", " + \
               str(self.pm1) + ", " + str(self.pm10) + ", " + str(self.pm25) + ")"

    def __iter__(self):
        for attr in self.attr_names:
            yield getattr(self, attr)

    @classmethod
    def attr_names(cls):
        names = [attr_name for attr_name in cls.__dict__ if '_' not in attr_name]
        if 'Sensor' in names:
            names.remove('Sensor')
        return names


class Sensor(Base):
    __tablename__ = 'sensors'
    __table_args__ = {"schema": "airbots"}

    sid = Column('sensor_id', Integer, primary_key=True)
    tid = Column('tile_id', Integer, ForeignKey('airbots.tiles.tile_id'), nullable=True)
    adr1 = Column('address1', String(50))
    adr2 = Column('address2', String(50))
    adrn = Column('address_num', String(5))
    lat = Column('latitude', Float)
    lon = Column('longitude', Float)
    elv = Column('elevation', Integer)
    measures = relationship('Measure', backref='Sensor', lazy='dynamic')

    def __init__(self, sensor_id=None, tile_id=None, address1=None, address2=None, address_num=None, latitude=None,
                 longitude=None, elevation=None):
        self.sid = sensor_id
        self.tid = tile_id
        self.adr1 = address1
        self.adr2 = address2
        self.adrn = address_num
        self.lat = latitude
        self.lon = longitude
        self.elv = elevation

    def __repr__(self):
        return "<Sensor(sensorid=%s,tileid=%s, lat=%s, lon=%s, elev=%s)>" % (self.sid, self.tid, self.lat, self.lon,
                                                                             self.elv)

    def __str__(self):
        return "Sensor_" + str(self.sid) + " Tile_" + str(self.tid) + " Address Line 1='" + str(
            self.adr1) + "' Address Line 2='" + str(self.adr2) + \
               "' Address Number='" + str(self.adrn) + "' lat,lon,elev=(" + str(self.lat) + ", " + \
               str(self.lon) + ", " + str(self.elv) + ")"

    def __iter__(self):
        for attr in self.attr_names:
            yield getattr(self, attr)

    @classmethod
    def attr_names(cls):
        return [attr_name for attr_name in cls.__dict__ if '_' not in attr_name]

    def nearestNeighbors(self, n):
        with Session as sesh:
            others = sesh.execute(select(Sensor).where(Sensor.sid != self.sid))

            distances = []

            startLL = MapPoint(self.lat, self.lon)
            for row in others:
                sensor = row[0]
                meters_away = round(calcDistance(startLL, MapPoint(sensor.lat, sensor.lon)), 3)
                distances.append((sensor, meters_away))

            if distances:
                distances.sort(key=lambda x: x[1])

            return distances[:n]

    def metersTo(self, other):
        if isinstance(other, Sensor):
            return calcDistance(startLL=MapPoint(self.lat, self.lon),
                                endLL=MapPoint(other.lat, other.lon))

    def getColumn(self, col, start_interval=datetime(2017, 11, 12, 0), end_interval=datetime(2021, 5, 5, 0)):
        with Session as sesh:
            return sesh.query(getattr(Measure, col)).filter(Measure.date >= start_interval).filter(
                Measure.date <= end_interval).where(Measure.sid == self.sid).all()

    def getMeasures(self, start_interval=datetime(2017, 11, 12, 0), end_interval=datetime(2021, 5, 5, 0)):
        with Session as sesh:
            return sesh.query(Measure).filter(Measure.date >= start_interval).filter(
                Measure.date <= end_interval).where(Measure.sid == self.sid).all()


class Tile(Base):
    __tablename__ = 'tiles'
    __table_args__ = {"schema": "airbots"}

    tid = Column('tile_id', Integer, primary_key=True)
    mid = Column('map_id', Integer, ForeignKey("airbots.maps.map_id"), nullable=False)
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

    def __iter__(self):
        for attr in self.attr_names:
            yield getattr(self, attr)

    @property
    def attr_names(self):
        return [attr_name for attr_name in self.__dict__ if '_' not in attr_name]

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
        self.updateClass(tile_class)

    def updateClass(self, tc):
        # session.query(). \
        #     filter(User.username == form.username.data). \
        #     update({"no_of_logins": (User.no_of_logins + 1)})
        # session.commit()
        with Session as sesh:
            sesh.execute(update(Tile).where(Tile.tid == self.tid).values(tclass=tc))
            sesh.commit()

    def set_vertices(self, vertex_list):
        if len(vertex_list) == self.numSides:
            for i in self.numSides:
                self.coordinates.append(vertex_list[i])

    def metersTo(self, other):
        if isinstance(other, Tile):
            start = MapPoint.createFromStr(self.center)
            end = MapPoint.createFromStr(other.center)
            return calcDistance(startLL=start, endLL=end)


class Map(Base):
    __tablename__ = 'maps'
    __table_args__ = {"schema": "airbots"}

    # collection of tiles-> collection tiles with coords and elevation.
    tileMesh = []
    aggregationOptions = []

    map_ID = Column('map_id', Integer, primary_key=True)
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


def getSensorsORM():
    with Session as sesh:
        return sesh.query(Sensor).all()


def getTilesORM(mapID=1):
    with Session as sesh:
        return sesh.query(Tile).where(Tile.mid == mapID).all()


def getTileORM(id):
    with Session as sesh:
        return sesh.query(Tile).where(Tile.tid == id).one()


def getSensorORM(id):
    with Session as sesh:
        return sesh.query(Sensor).where(Sensor.sid == id).one()


def getMeasureORM(sid, date):
    with Session as sesh:
        return sesh.query(Measure).where(Measure.date == date).where(Measure.sid == sid).one()


def getMeasuresORM(sid, start_interval=datetime(2018, 9, 3, 0), end_interval=datetime(2021, 5, 5, 0)):
    with Session as sesh:
        return sorted(sesh.query(Measure).filter(Measure.date >= start_interval).filter(
            Measure.date <= end_interval).where(Measure.sid == sid).all(), key=lambda x: x.dk)


def getOtherSensorsORM(sid):
    with Session as sesh:
        return sesh.query(Sensor).where(Sensor.sid != sid).all()


def createAllTables(eng):
    table_objects = [Map.__table__, Tile.__table__, Sensor.__table__, Measure.__table__]
    Base.metadata.create_all(eng, tables=table_objects)


def sensorMerge(fname):
    f = open(fname).read().splitlines()
    merged_sensors = []
    merged_measures = []
    conn = createConnection()
    for line in f:
        sensors = re.split('\+', line)
        if len(sensors) == 1:
            merged_sensors.append(getSensor(conn, sensors[0]))
            merged_measures.extend(getMeasures(conn, sensors[0]))
        else:
            print("Merging:", sensors)
            temp_list = []
            if '*' in sensors[0]:
                sid = re.split('\*', sensors[0])
                temp = getSensor(conn, sid[0])
                merged_sensors.append(temp)
                temp_list.extend(getMeasures(conn, sid[0]))
                print(temp_list.pop())
                temp_list.extend(updateMeasures(getMeasures(conn, sensors[1]), sid[0]))
            else:
                sid = re.split('\*', sensors[1])
                temp = getSensor(conn, sid[0])
                merged_sensors.append(temp)
                temp_list.extend(updateMeasures(getMeasures(conn, sensors[0]), sid[0]))
                print(temp_list.pop())
                temp_list.extend(getMeasures(conn, sid[0]))

            merged_measures.extend(temp_list)
    return merged_sensors, merged_measures


def updateMeasures(m_list, sid):
    for m in m_list:
        m.sensorid = sid
    return m_list


def getObservations(exclude=None):
    attributes = [attr_name for attr_name in Measure.__dict__ if not str(attr_name).startswith("_")]
    attributes.remove('dk')
    attributes.remove('Sensor')
    attributes.remove('date')
    if exclude in attributes:
        attributes.remove(exclude)

    return attributes


def populateTables():
    conn = createConnection()
    s_list, m_list = sensorMerge(r"C:\Users\mrusieck\PycharmProjects\AirBot\docs\Sensor_Merge")
    print("sensor data fetched")
    print("measurement data fetched")
    # tiles = getTiles(conn, '*')
    # print("tilebin data fetched")
    conn.close()
    # print("inserting maps..")
    # addOpoleMap()
    # print("inserting tiles..")
    # insertTiles(tiles)
    print("inserting sensors..")
    insertSensors(s_list)
    print("inserting measures..")
    insertMeasures(m_list)
