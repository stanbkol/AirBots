from edu.pwr.database.DataProcessing import *
from edu.pwr.database.DbManager import insertSensors, insertMeasures, addOpoleMap
import psycopg2
import matplotlib.pyplot as plt
from edu.pwr.airbots.wma import *
from IPython.display import display
from edu.pwr.database.DbManager import engine, Base, Session, insertTiles
from edu.pwr.map.Map import Map
from edu.pwr.map.Sensor import Sensor
from edu.pwr.map.Measure import Measure
from edu.pwr.map.Tile import Tile


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


def show_wma():
    conn = createConnection()
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 8, 0)
    cols, measures = getPM1(conn, 11563, start, end)
    conn.close()
    res = weighted_rolling_mean(measures, 24, cols, exp=True)
    plt.plot(res['date'], res['pm1'], label="PM1 Values")
    plt.plot(res['date'], res['MA_PM1'], label="MA Values")
    plt.xlabel('Dates')
    plt.ylabel('Values')
    plt.legend()
    plt.show()


def populateTables():
    conn = createConnection()
    s_list = getSensors(conn, '*')
    print("sensor data fetched")
    m_list = getMeasures(conn, '*')
    print("measurement data fetched")
    tiles = getTiles(conn, '*')
    print("tilebin data fetched")
    conn.close()
    print("inserting maps..")
    addOpoleMap()
    print("inserting tiles..")
    insertTiles(tiles)
    print("inserting sensors..")
    insertSensors(s_list)
    print("inserting measures..")
    insertMeasures(m_list)


def main():
    # populateTables()
    # createAirbots(eng)
    # print(eng)
    print("creating tables..")
    Base.metadata.create_all(engine)
    populateTables()


if __name__ == '__main__':
    main()
