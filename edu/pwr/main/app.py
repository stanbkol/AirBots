from edu.pwr.database.DataProcessing import *
import psycopg2
import matplotlib.pyplot as plt
from edu.pwr.airbots.wma import *
from IPython.display import display


def getSensor(conn, sid):
    with conn.cursor() as cursor:
        query = 'SELECT * FROM dbo.Sensors WHERE sensorID = %s;'
        data = [sid]
        cursor.execute(query, data)
        sensor_info = cursor.fetchone()
        return Sensor(sensor_info[0], sensor_info[1], sensor_info[2], sensor_info[3], sensor_info[4], sensor_info[5], sensor_info[6], sensor_info[7])


def getPM1(conn, sid, start_interval=None, end_interval=None):
    with conn.cursor() as cursor:
        if not start_interval and not end_interval:
            query = 'SELECT * FROM dbo.Measurements WHERE sensorID = %s;'
            data = [sid]
        else:
            query = 'SELECT sensorid, date, pm1 FROM dbo.Measurements WHERE sensorID = %s AND date BETWEEN %s AND %s;'
            data = [sid, start_interval, end_interval]
        cursor.execute(query, data)
        cols = []
        for elt in cursor.description:
            cols.append(elt[0])
        data_list = cursor.fetchall()
        return cols, data_list

def main():
    print("connecting to server")
    conn = createConnection()
#     dropTiles(conn)
#     createTilesTable(conn)
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 8, 0)
    # my_sensor = getSensor(conn, 11563)
    cols, measures = getPM1(conn, 11563, start, end)
    conn.close()
    res = weighted_rolling_mean(measures, 24, cols, exp=False)
    print(res)

    plt.plot(res['date'], res['pm1'], label="PM1 Values")
    plt.plot(res['date'], res['MA'], label="MA Values")
    # plt.plot(date_results, pm10_results, label="PM10 Values")
    # plt.plot(date_results, pm25_results, label="PM25 Values")
    # plt.plot(date_results, temp_results, label="Temperature Values")
    plt.xlabel('Dates')
    plt.ylabel('Values')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
