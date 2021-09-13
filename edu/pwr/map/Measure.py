from sqlalchemy import create_engine, Column, String, Integer, Float, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base

base = declarative_base()


class Measure(base):
    __tablename__ = 'measures'
    __table_args__ = {"schema": "airbots"}

    sid = Column('sensor_id', Integer,  primary_key=True)
    date = Column('date', DateTime,  primary_key=True)
    temp = Column('temperature', Float)
    pm1 = Column('pm1', Float)
    pm10 = Column('pm10', Float)
    pm25 = Column('pm25', Float)

    def __init__(self, d, s_id, pm1, pm25, pm10, t):
        self.date = d
        self.sid = s_id
        self.temp = t
        self.pm1 = pm1
        self.pm10 = pm10
        self.pm25 = pm25