import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateSchema
from edu.pwr.map.Sensor import Sensor
from edu.pwr.map.Map import Map
from edu.pwr.map.Measure import Measure
from sqlalchemy.ext.declarative import declarative_base



# def createSensors():
#     eng = createEngine()
#     sensor = Sensor()
#     sensor.create(eng)


def createMaps():
    eng = createEngine()
    new_map = Map()
    new_map.create(eng)


def createMeasures():
    eng = createEngine()
    measure = Measure()
    measure.create(eng)


def createSession(engine):
    Session = sessionmaker(engine)
    return Session()


def createEngine(dialect="postgresql", driver=None, db_user="asds_PWR", password="W#4bvgBxDi$v6zB",
                 host="pgsql13.asds.nazwa.pl", database="asds_PWR"):
    if driver:
        db_string = f'{dialect}+{driver}://{db_user}:{password}@{host}/{database}'
    else:
        db_string = f'{dialect}://{db_user}:{password}@{host}/{database}'

    print(db_string)
    return create_engine(db_string)


def createAirbots(eng):
    eng.execute(CreateSchema('airbots'))


def createSensors(eng):
    Base = declarative_base()
    table_objects = [Sensor.__table__]
    Base.metadata.create_all(eng, tables=table_objects)


