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


def insertSensors(sensors):
    from edu.pwr.map.Sensor import Sensor

    with Session as session:
        orm_sensors = [
            Sensor(sensor_id=s.sensorid, tile_id=s.tileid, address1=s.address1, address2=s.address2, address_num=s.addressnumber,
                   latitude=s.latitude, longitude=s.longitude, elevation=s.elevation)
            for s in sensors
        ]
        session.bulk_save_objects(orm_sensors)
        session.commit()
    print("Sensor inserts completed")


def insertMeasures(measures):
    from edu.pwr.map.Measure import Measure

    with Session as session:
        orm_measures = [Measure(date_key=m.datekey, sensor_id=m.sensorid, date=m.date, pm1=m.pm1, pm25=m.pm25, pm10=m.pm10,
                              temperature=m.temperature) for m in measures]
        session.bulk_save_objects(orm_measures)
        session.commit()
    print("Measurement inserts completed")


def insertTiles(tilebins):
    from edu.pwr.map.Tile import Tile

    with Session as session:
        tiles = [
            Tile(tileID=tb.tileid, mapID=tb.mapid, numSides=tb.numSides, coordinates=tb.coordinates,
                 diameter=tb.diameter, center=tb.centerlatlon, tileClass=tb.tclass, max_elevation=tb.max_elevation,
                 min_elevation=tb.min_elevation, temperature=tb.temperature,
                 pm10_avg=tb.pm10_avg, pm1_avg=tb.pm1_avg, pm25_avg=tb.pm25_avg)
            for tb in tilebins
        ]
        session.bulk_save_objects(tiles)
        session.commit()
    print("Tile inserts completed")



def addOpoleMap():
    with Session as sesh:
        from edu.pwr.map.Map import Map

        opole = Map(map_ID=1, name="Opole", coord_NW="50.76997429,17.77959063", coord_NE="50.76997429,18.03269049",
                    coord_SE="50.58761735,18.03269049", coord_SW="50.58761735,17.77959063")

        sesh.add(opole)
        sesh.commit()
