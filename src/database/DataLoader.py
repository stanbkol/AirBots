from datetime import datetime

from src.database.oldEntry import Entry
from src.database.oldSensor import Sensor
import psycopg2

from src.map.MapPoint import MapPoint, calcDistance
from src.map.TileBin import TileBin


class Loader:
    def __init__(self):
        pass


def createSchema(name, conn):
    with conn.cursor() as cursor:
        cursor.execute('''CREATE SCHEMA %s;''' % (name))


def dropTables(conn):
    with conn.cursor() as cursor:
        cursor.execute('''DROP TABLE dbo.Sensors;''')
        cursor.execute('''DROP TABLE dbo.Measurements;''')
        # cursor.execute('''DROP TABLE dbo.Tiles ;''')
        conn.commit()


def dropTiles(conn):
    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE dbo.Tiles;")
        conn.commit()


def createTilesTable(conn):
    with conn.cursor() as cursor:
        print("\t creating tiles")
        cursor.execute('''CREATE TABLE dbo.Tiles (tileId int primary key,
                                                mapId int,
                                                class varchar(50),
                                                diameter float,
                                                centerLatLon varchar(50),
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
                                                polygon varchar(500)
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
                       "VALUES(%s, %s, %s, %s, %s, %s, %s)", (
                       dk, measure.sensorid, measure.date, measure.pm1, measure.pm25, measure.pm10,
                       measure.temperature))
        conn.commit()


def insertSensor(conn, sensor):
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO dbo.Sensors (sensorID, tileId, address1, address2, addressNumber, latitude, longitude, elevation) "
            "VALUES(%s, %s, %s, %s, %s, %s, %s, %s)",
            (int(sensor.sensorid), sensor.tileid, sensor.address1, sensor.address2, sensor.addressnumber,
             sensor.latitude, sensor.longitude, int(sensor.elevation)))
        conn.commit()


def insertTile(conn, tile):
    with conn.cursor() as cursor:
        insert_sql = '''INSERT INTO dbo.Tiles (tileId,
                                                mapId,
                                                class,
                                                diameter,
                                                centerLatLon,
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
                                                polygon
                                                ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'''

        cursor.execute(insert_sql, (int(tile.tileid), int(tile.mapid), tile.tclass, tile.diameter,
                                    tile.centerlatlon.latlon_str,
                                    tile.coordinates[0].latlon_str, tile.coordinates[1].latlon_str,
                                    tile.coordinates[2].latlon_str,
                                    tile.coordinates[3].latlon_str, tile.coordinates[4].latlon_str,
                                    tile.coordinates[5].latlon_str,
                                    tile.min_elevation, tile.max_elevation,
                                    tile.temperature, tile.pm10_avg, tile.pm1_avg, tile.pm25_avg, tile.poly_str))

        conn.commit()


def fetchValidSensors(conn):
    with conn.cursor() as cursor:
        fetch_sql = '''SELECT * FROM dbo.Sensors
                        WHERE latitude != -1 AND longitude != -1;'''
        cursor.execute(fetch_sql)
        return cursor.fetchall()


def fetchMapGridPolys(conn, mapId):
    with conn.cursor() as cursor:
        fetch_sql = '''SELECT vertex1, vertex2, vertex3, vertex4, vertex5, vertex6 FROM dbo.Tiles WHERE MapId = %s;'''
        cursor.execute(fetch_sql, (mapId,))
        return cursor.fetchall()


def update_sensor_tile(conn, sensor_id, tile_id):
    with conn.cursor() as cursor:
        sql = "UPDATE dbo.Sensors SET tileid = %s WHERE sensorid = %s"
        cursor.execute(sql, (tile_id, sensor_id))
        conn.commit()


def getSensorsAll(conn):
    with conn.cursor() as cursor:
        query1 = 'SELECT * FROM dbo.Sensors;'
        cursor.execute(query1)
        return cursor.fetchall()


def getMeasures(conn, *field_names, chunk_size=2000):
    if '*' in field_names:
        fields_format = '*'
        field_names = [field.name for field in Entry.get_db_fields(conn)]
    else:
        fields_format = ', '.join(field_names)

    query = f"SELECT {fields_format} FROM dbo.measurements"

    with conn.cursor() as cursor:
        cursor.execute(query)

        measurement_objects = list()
        fetching_completed = False
        while not fetching_completed:
            rows = cursor.fetchmany(size=chunk_size)
            for row in rows:
                row_data = dict(zip(field_names, row))
                measurement_objects.append(Entry.entry_set_fields(**row_data))
            fetching_completed = len(rows) < chunk_size
        return measurement_objects


def getSensors(conn, *field_names, chunk_size=2000):
    if '*' in field_names:
        fields_format = '*'
        field_names = [field.name for field in Sensor.get_db_fields(conn)]
    else:
        fields_format = ', '.join(field_names)

    query = f"SELECT {fields_format} FROM dbo.sensors"

    with conn.cursor() as cursor:
        cursor.execute(query)

        sensor_objects = list()
        fetching_completed = False
        while not fetching_completed:
            rows = cursor.fetchmany(size=chunk_size)
            for row in rows:
                row_data = dict(zip(field_names, row))
                sensor_objects.append(Sensor.sensor_set_fields(**row_data))

            fetching_completed = len(rows) < chunk_size

        return sensor_objects


def getTiles(conn, *field_names, chunk_size=2000):
    print("fetching tilebins")
    if '*' in field_names:
        fields_format = '*'
        field_names = [field.name for field in TileBin.get_db_fields(conn)]
    else:
        fields_format = ', '.join(field_names)

    query = f"SELECT {fields_format} FROM dbo.tiles"

    with conn.cursor() as cursor:
        cursor.execute(query)

        tile_objects = list()
        fetching_completed = False
        while not fetching_completed:
            rows = cursor.fetchmany(size=chunk_size)
            for row in rows:
                row_data = dict(zip(field_names, row))
                tile_objects.append(TileBin.tilebin_set_fields(**row_data))

            fetching_completed = len(rows) < chunk_size

        return tile_objects


def getOtherSensors(conn, exclude_id, batch_size=2000):
    field_names = [field.name for field in Sensor.get_db_fields(conn)]

    with conn.cursor() as cursor:
        sql = '''SELECT * from dbo.sensors WHERE sensorid != %s'''
        cursor.execute(sql, (exclude_id,))

        sensor_objects = list()
        fetching_completed = False
        while not fetching_completed:
            rows = cursor.fetchmany(size=batch_size)
            for row in rows:
                row_data = dict(zip(field_names, row))
                sensor_objects.append(Sensor.sensor_set_fields(**row_data))

            fetching_completed = len(rows) < batch_size

        return sensor_objects


def findNearestSensors(conn, sensorid):
    base_sensor = getSensor(conn, sensorid)

    sensors = getOtherSensors(conn, sensorid)
    distances = []
    startLL = MapPoint(base_sensor.latitude, base_sensor.longitude)
    for sensor in sensors:
        meters_away = calcDistance(startLL, MapPoint(sensor.latitude, sensor.longitude))
        distances.append((sensor, meters_away))

    distances.sort(key=lambda x: x[1])

    return distances


def getSensor(conn, sid):
    with conn.cursor() as cursor:
        query = 'SELECT * FROM dbo.Sensors WHERE sensorID = %s;'
        data = [sid]
        cursor.execute(query, data)
        sensor_info = cursor.fetchone()
        return Sensor(sensor_info[0], sensor_info[1], sensor_info[2], sensor_info[3], sensor_info[4], sensor_info[5],
                      sensor_info[6], sensor_info[7])


def getMeasures(conn, sid):
    with conn.cursor() as cursor:
        query = 'SELECT * FROM dbo.Measurements WHERE sensorID = %s;'
        data = [sid]
        cursor.execute(query, data)
        measures = cursor.fetchall()
        m_list = []
        for measure_info in measures:
            m_list.append(Entry(measure_info[0], measure_info[1], measure_info[2], measure_info[3], measure_info[4],
                                measure_info[5],
                                measure_info[6]))
        return m_list


def createConnection():
    conn = psycopg2.connect(
        host="pgsql13.asds.nazwa.pl",
        database="asds_PWR",
        user="asds_PWR",
        password="W#4bvgBxDi$v6zB")
    return conn
