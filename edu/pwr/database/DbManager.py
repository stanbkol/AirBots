import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.schema import CreateSchema
from edu.pwr.database.DataLoader import getSensors, getTiles, createConnection

def createSession(engine):
    Session = sessionmaker(bind=engine)
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


# def createAllTables(eng):
#     table_objects = [Map.__table__, Tile.__table__, Sensor.__table__, Measure.__table__]
#     Base.metadata.create_all(eng, tables=table_objects)


engine = createEngine()
Session = createSession(engine)
Base = declarative_base(bind=engine)


def insertTiles(tilebins):
    from edu.pwr.map.Tile import Tile

    with Session as session:
        for tb in tilebins:
            tile = Tile(tileID=tb.tileid, mapID=tb.mapid, numSides=tb.numSides, coordinates=tb.coordinates,
                        diameter=tb.diameter, center=tb.centerlatlon, tileClass=tb.tclass, max_elevation=tb.max_elevation,
                        min_elevation=tb.min_elevation, temperature=tb.temperature,
                        pm10_avg=tb.pm10_avg, pm1_avg=tb.pm1_avg, pm25_avg=tb.pm25_avg)
            print(repr(tile))
            session.add(tile)
            session.commit()


