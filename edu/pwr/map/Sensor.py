from sqlalchemy import create_engine, Column, String, Integer, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from edu.pwr.database.DbManager import createEngine

base = declarative_base()


class Sensor(base):
    __tablename__ = 'sensors'

    sid = Column('sensor_id', Integer, primary_key=True)
    tid = Column('tile_id', Integer, ForeignKey("user.user_id"), nullable=False)
    adr1 = Column('address1', String(50))
    adr2 = Column('address2', String(50))
    adrn = Column('address_num', String(5))
    lat = Column('latitude', Float)
    lon = Column('longitude', Float)
    elv = Column('elevation', Integer)

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

    def __str__(self):
        return "Sensor_%s latlon,elv: (%s, %s, %s)" % (self.sid, self.lat, self.lon, self.elv)
