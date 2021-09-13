from sqlalchemy import create_engine, Column, String, Integer, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from edu.pwr.database.DbManager import Base


class Measure(Base):
    __tablename__ = 'measures'
    __table_args__ = {"schema": "airbots"}

    datekey = Column('date_key', Integer, primary_key=True)
    sid = Column('sensor_id', Integer, ForeignKey("airbots.sensors.sensor_id"), primary_key=True)
    date = Column('date', DateTime)
    temp = Column('temperature', Float)
    pm1 = Column('pm1', Float)
    pm10 = Column('pm10', Float)
    pm25 = Column('pm25', Float)

    sensors = relationship("Sensor", secondary="airbots.sensors")


    def __init__(self, date_key, s_id, date, pm1, pm25, pm10, temperature):
        self.datekey = date_key
        self.sid = s_id
        self.date = date
        self.temp = temperature
        self.pm1 = pm1
        self.pm10 = pm10
        self.pm25 = pm25