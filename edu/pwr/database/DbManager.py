import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.schema import CreateSchema


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


def insertSensors(sensors):
    from edu.pwr.map.Sensor import Sensor

    with Session as session:
        for s in sensors:
            sensor = Sensor(sensor_id=s.sid, tile_id=s.tid, address1=s.adr1, address2=s.adr2, address_num=s.adrn,
                            latitude=s.lat,
                            longitude=s.long, elevation=s.elv)
            session.add(sensor)
            session.commit()
    print("Sensor Inserts Complete")


def insertMeasures(measures):
    from edu.pwr.map.Measure import Measure

    with Session as session:
        for m in measures:
            measure = Measure(date_key=m.dk, sensor_id=m.sid, date=m.date, pm1=m.pm1, pm25=m.pm25, pm10=m.pm10,
                              temperature=m.temp)
            session.add(measure)
            session.commit()
    print("Measurement Inserts Complete")


def insertTiles(sesh, tilebins):
    from edu.pwr.map.Tile import Tile

    with sesh as session:
        for tb in tilebins:
            tile = Tile(tileID=tb.tileId, mapID=tb.mapId, numSides=tb.numSides, coordinates=tb.coordinates,
                        diameter=tb.diameter, center=tb.centerPt, tileClass=tb.tileClass, max_elevation=tb.max_elvation,
                        min_elevation=tb.min_elevation, temperature=tb.temperature,
                        pm10_avg=tb.pm10_avg, pm1_avg=tb.pm1_avg, pm25_avg=tb.pm25)

            session.add(tile)
            session.commit()
