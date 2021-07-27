from datetime import datetime

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
        cursor.commit()


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
        cursor.commit()


def createMeasurementsTable(conn):
    with conn.cursor() as cursor:
        print("\tmeasurements")
        cursor.execute('''CREATE TABLE Measurements (dateKey int,
                                                     sensorID int,
                                                     timestamp datetime,
                                                     pm1 float,
                                                     pm25 float,
                                                     pm10 float,
                                                     temperature float,
                                                     CONSTRAINT pk_measures PRIMARY KEY (dateKey, sensorID)
                                                     );
                        ''')
        cursor.commit()


def insertMeasure(measure, conn):
    rawDate = datetime.strptime(measure.date, '%m/%d/%Y %H:%M')
    dk = int(rawDate.strftime('%Y%m%d%H'))
    with conn.cursor() as cursor:
        insert_sql = '''INSERT INTO Measurements (dateKey, 
                                                    sensorID, 
                                                    timestamp, 
                                                    pm1, 
                                                    pm25, 
                                                    pm10, 
                                                    temperature) 
                                                    VALUES(?,?,?,?,?,?,?)'''
        cursor.execute(insert_sql, dk, measure.SID, measure.date, measure.pm1, measure.pm25, measure.pm10, measure.temp)
        cursor.commit()


def insertSensor(conn, sensor):

    with conn.cursor() as cursor:
        insert_sql = '''INSERT INTO dbo.Sensors (sensorID, tileId, address1, address2, addressNumber, latitude, longitude, elevation)
                            VALUES (?,?,?,?,?,?,?,?)'''
        cursor.execute(insert_sql, int(sensor.SID),
                                    int(sensor.tile),
                                    sensor.address_1,
                       sensor.address_2, sensor.address_num, sensor.latitude, sensor.longitude, int(sensor.elevation))
        cursor.commit()
