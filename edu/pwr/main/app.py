from edu.pwr.database.DataProcessing import *
import psycopg2
import matplotlib.pyplot as plt
from edu.pwr.airbots.wma import *
from IPython.display import display
from edu.pwr.database.DbManager import *


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


def main():
    print("connecting to server")

    # db_str = "postgresql://asds_PWR:W#4bvgBxDi$v6zB@pgsql13.asds.nazwa.pl/asds_PWR"
    # eng = createEngine(db_string)
    eng = createEngine()
    createAirbots(eng)
    print(eng)
    print("creating sensors..")
    createSensors(eng)


if __name__ == '__main__':
    main()
