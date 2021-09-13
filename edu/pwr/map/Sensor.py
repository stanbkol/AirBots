from sqlalchemy import Column, String, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from edu.pwr.database.DbManager import Base


from edu.pwr.map.MapPoint import calcDistance, MapPoint


class Sensor(Base):
    __tablename__ = 'sensors'
    __table_args__ = {"schema": "airbots"}

    sid = Column('sensor_id', Integer, primary_key=True)
    tid = Column('tile_id', Integer, ForeignKey('airbots.tiles.tile_id'), nullable=False)
    adr1 = Column('address1', String(50))
    adr2 = Column('address2', String(50))
    adrn = Column('address_num', String(5))
    lat = Column('latitude', Float)
    lon = Column('longitude', Float)
    elv = Column('elevation', Integer)

    tiles = relationship("Tile", secondary="airbots.tiles")

    def __init__(self, sensorid=None, tid=None, address1=None, address2=None, address_num=None, latitude=None,
                 longitude=None, elevation=None):
        self.sid = sensorid
        self.tid = tid
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

    def setAgent(self, a):
        self.agent = a

    def changeState(self, s):
        self.state = s

    def __str__(self):
        return "Sensor_" + str(self.sensorid) + " Address Line 1='" + str(self.address1) + "' Address Line 2='" + str(self.address2) + \
               "' Address Number='" + str(self.addressnumber) + "' lat,lon,elev=(" + str(self.latitude) + ", " + \
               str(self.longitude) + ", " + str(self.elevation) + ")"

    def metersTo(self, other):
        if isinstance(other, Sensor):
            return calcDistance(startLL=MapPoint(self.latitude, self.longitude),
                                endLL=MapPoint(other.latitude, other.longitude))
