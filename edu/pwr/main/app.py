from edu.pwr.database.DataProcessing import *
import psycopg2


def getSensor(conn, sid):
    with conn.cursor() as cursor:
        query = 'SELECT * FROM dbo.Sensors WHERE sensorID = %s;'
        data = [sid]
        cursor.execute(query, data)
        sensor_info = cursor.fetchone()
        return Sensor(sensor_info[0], sensor_info[1], sensor_info[2], sensor_info[3], sensor_info[4], sensor_info[5], sensor_info[6], sensor_info[7])


def main():
    print("connecting to server")
    conn = createConnection()
#     dropTiles(conn)
#     createTilesTable(conn)
    start = datetime(2019, 12, 1, 0)
    end = datetime(2019, 12, 31, 23)
    my_sensor = getSensor(conn, 11563)
    my_sensor.getData(conn, start, end)
    #dataSummary(conn, start, end)
    conn.close()


if __name__ == '__main__':
    main()
