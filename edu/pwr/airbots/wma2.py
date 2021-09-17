from edu.pwr.database.DataProcessing import *
import psycopg2
import matplotlib.pyplot as plt
from edu.pwr.airbots.wma import *
from IPython.display import display


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
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 5, 31, 0)
    cols, measures = getPM1(conn, 11563, start, end)
    df = pd.DataFrame(data=measures, columns=cols)
    conn.close()
    res = df.sort_values(by="date", ascending=True)
    res['Prediction'] = res.pm1.rolling(10, min_periods=1).mean()
    res['Error'] = abs(((res['pm1']-res['Prediction'])/res['pm1'])*100)
    #rolling(x, min periods=y): higher x, lower accuracy
    print(res)
    plt.plot(res['date'], res['pm1'], label="PM1 Values")
    plt.plot(res['date'], res['Prediction'], label="Pred")
     # plt.plot(df2['date'], df2['pm1'], label="Prediction")
    plt.xlabel('Dates')
    plt.ylabel('Values')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
