from datetime import datetime

import psycopg2
import pyodbc


class Loader:
    def __init__(self):
        pass


def createSchema(name, conn):
    with conn.cursor() as cursor:
        cursor.execute('''CREATE SCHEMA %s;''' % (name))


def dropTables(conn):
    with conn.cursor() as cursor:
        cursor.execute('''DROP TABLE dbo.Sensors;''')
        cursor.execute('''DROP TABLE dbo.Measurements ;''')
        #cursor.execute('''DROP TABLE dbo.Tiles ;''')
        conn.commit()


def createTilesTable(conn):
    with conn.cursor() as cursor:
        print("\tsensors")
        cursor.execute('''CREATE TABLE dbo.Tiles (TileID int primary key,
                                                MapId int,
                                                class varchar(25),
                                                diameter float,
                                                centerLatLong varchar(50),
                                                vertex1 varchar(50),
                                                vertex2 varchar(50),
                                                vertex3 varchar(50),
                                                vertex4 varchar(50),
                                                vertex5 varchar(50),
                                                vertex6 varchar(50),
                                                min_elevation float,
                                                max_elevation float,
                                                temperature float,
                                                pm10_avg float, 
                                                pm1_avg float,
                                                pm25_avg float,
                                                );
                            ''')
        conn.commit()


def createSensorsTable(conn):
    with conn.cursor() as cursor:
        print("\tsensors")
        cursor.execute('''CREATE TABLE dbo.Sensors (sensorID int primary key,
                                                tileId int, 
                                                address1 varchar(50),
                                                address2 varchar(50),
                                                addressNumber varchar(5),
                                                latitude float,
                                                longitude float,
                                                elevation int
                                                );
                            ''')
        conn.commit()


def createMeasurementsTable(conn):
    with conn.cursor() as cursor:
        print("\tmeasurements")
        cursor.execute('''CREATE TABLE dbo.Measurements (dateKey int,
                                                     sensorID int,
                                                     date timestamp,
                                                     pm1 float,
                                                     pm25 float,
                                                     pm10 float,
                                                     temperature float,
                                                     CONSTRAINT pk_measures PRIMARY KEY (dateKey, sensorID)
                                                     );
                        ''')
        conn.commit()


def insertMeasure(measure, conn):
    rawDate = datetime.strptime(measure.date, '%m/%d/%Y %H:%M')
    dk = int(rawDate.strftime('%Y%m%d%H'))
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO dbo.Measurements (dateKey, sensorID, date, pm1, pm25, pm10, temperature) "
            "VALUES(%s, %s, %s, %s, %s, %s, %s)", (dk, measure.SID, measure.date, measure.pm1, measure.pm25, measure.pm10, measure.temp))
        conn.commit()


def insertSensor(conn, sensor):
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO dbo.Sensors (sensorID, tileId, address1, address2, addressNumber, latitude, longitude, elevation) "
                       "VALUES(%s, %s, %s, %s, %s, %s, %s, %s)", (int(sensor.SID), sensor.tile, sensor.address_1, sensor.address_2, sensor.address_num,sensor.latitude, sensor.longitude, int(sensor.elevation)))
        conn.commit()


def insertTile(conn, tile):
    with conn.cursor() as cursor:
        insert_sql = '''INSERT INTO dbo.Tiles (TileID,
                                                MapId,
                                                class,
                                                diameter,
                                                centerLatLong,
                                                vertex1,
                                                vertex2,
                                                vertex3,
                                                vertex4,
                                                vertex5,
                                                vertex6,
                                                min_elevation,
                                                max_elevation,
                                                temperature,
                                                pm10_avg, 
                                                pm1_avg,
                                                pm25_avg,
                                                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''

        cursor.execute(insert_sql, int(tile.tid), int(tile.mapId), tile.tileClass, tile.diameter, tile.center, tile.coordinates[0],
                       tile.coordinates[1], tile.coordinates[2], tile.coordinates[3], tile.coordinates[4], tile.coordinates[5], tile.min_elevevation, tile.max_elevation,
                       tile.temperature, tile.pm10_avg, tile.pm1_avg, tile.pm25_avg)

        cursor.commit()


def fetchValidSensors(conn):
    with conn.cursor() as cursor:
        fetch_sql = '''SELECT * FROM dbo.Sensors
                        WHERE latitude!=-1 AND longitude!=-1;'''
        cursor.execute(fetch_sql)
        return cursor.fetchall()


def createConnection(user=None, pwrd=None, server=None):
    conn = psycopg2.connect(
        host="pgsql13.asds.nazwa.pl",
        database="asds_PWR",
        user="asds_PWR",
        password="W#4bvgBxDi$v6zB")
    return conn
