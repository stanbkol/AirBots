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

