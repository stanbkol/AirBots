from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.future import select
from sqlalchemy.orm import relationship
from src.database.DbManager import Base, Session
from src.map.MapPoint import calcDistance, MapPoint


class Sensor(Base):
    __tablename__ = 'sensors'
    __table_args__ = {"schema": "agents"}

    sid = Column('sensor_id', Integer, primary_key=True)
    tid = Column('tile_id', Integer, ForeignKey('agents.tiles.tile_id'), nullable=True)
    adr1 = Column('address1', String(50))
    adr2 = Column('address2', String(50))
    adrn = Column('address_num', String(5))
    lat = Column('latitude', Float)
    lon = Column('longitude', Float)
    elv = Column('elevation', Integer)
    measures = relationship('Measure', backref='Sensor', lazy='dynamic')
    # tiles = relationship("Tile")

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
        self.agent = None
        self.state = True

    def __repr__(self):
        return "<Sensor(sensorid=%s,tileid=%s, lat=%s, lon=%s, elev=%s)>" % (self.sid, self.tid, self.lat, self.lon,
                                                                             self.elv)

    def __str__(self):
        return "Sensor_" + str(self.sid) + " Address Line 1='" + str(self.adr1) + "' Address Line 2='" + str(self.adr2) + \
               "' Address Number='" + str(self.adrn) + "' lat,lon,elev=(" + str(self.lat) + ", " + \
               str(self.lon) + ", " + str(self.elv) + ")"

    @classmethod
    def getSensor(cls, id):
        with Session as sesh:
            return sesh.query(Sensor).get(id)

    def setAgent(self, a):
        self.agent = a

    def changeState(self, s):
        self.state = s

    def metersTo(self, other):
        if isinstance(other, Sensor):
            return calcDistance(startLL=MapPoint(self.latitude, self.longitude),
                                endLL=MapPoint(other.latitude, other.longitude))

    def nearest_neighbors(self, n):
        with Session as sesh:
            others = sesh.execute(select(Sensor).where(Sensor.sid != self.sid))

            distances = []

            startLL = MapPoint(self.lat, self.lon)
            for row in others:
                sensor = row[0]
                meters_away = round(calcDistance(startLL, MapPoint(sensor.lat, sensor.lon)),3)
                distances.append((sensor, meters_away))

            if distances:
                distances.sort(key=lambda x: x[1])

            return distances[:n]

