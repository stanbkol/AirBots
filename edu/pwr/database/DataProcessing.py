import unicodedata
from datetime import time
import psycopg2
<<<<<<< HEAD:edu/pwr/database/DataProcessing.py
from edu.pwr.database.DataLoader import *
from edu.pwr.database.utils import *
from edu.pwr.database.Entry import *
from edu.pwr.database.Sensor import *
=======
import matplotlib.pyplot as plt
from DataLoader import *

from utils import *

from Entry import *
from Sensor import *
>>>>>>> 9ef7d316fdb62ea8147855a9cc36409f4adc2274:edu/pwr/database/main.py

invalid_count = 0
sensor_list = []
dict_list = {}
final_list = []


# def calcDaily(SID, date, entries):
#     avg_temp = np.mean([e.temp for e in entries])
#     avg_humid = np.mean([e.humidity for e in entries])
#     avg_press = np.mean([e.pressure for e in entries])
#     avg_pm1 = np.mean([e.pm1 for e in entries])
#     avg_pm25 = np.mean([e.pm25 for e in entries])
#     avg_pm10 = np.mean([e.pm10 for e in entries])
#     return Entry(date, SID, avg_temp, avg_humid, avg_press, avg_pm1, avg_pm25, avg_pm10)
#
#
# def writeAvgs(filename):
#     columns = ['SID', 'Date', 'Temperature', 'Humidity', 'Pressure', 'Pm1', 'Pm10', 'Pm25']
#     with open(filename, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(columns)
#         for e in final_list:
#             data = [e.SID, e.date, e.temp, e.humidity, e.pressure, e.pm1, e.pm10, e.pm25]
#             writer.writerow(data)
#         data = ["Invalid Count", invalid_count]
#         writer.writerow(data)
#     f.close()
#
# def calcAverages():
#     global dict_list
#     global final_list
#     for sensor in dict_list:
#         for entry in dict_list[sensor]:
#             date_list = []
#             temp = entry.date.split("T")
#             date = temp[0]
#             hour = temp[1].split(":")[0]
#             date_list.append(entry)
#             if hour == "23":
#                 final_list.append(calcDaily(sensor, date, date_list))
#                 date_list.clear()
# def writeEntries(filename):
#     columns = ['SID', 'Date', 'Temperature', 'Humidity', 'Pressure', 'Pm1', 'Pm10', 'Pm25']
#     with open(filename, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(columns)
#         for e in entry_list:
#             data = [e.SID, e.date, e.temp, e.humidity, e.pressure, e.pm1, e.pm10, e.pm25]
#             writer.writerow(data)
#         writer.writerow(data)
#     f.close()
# def popDictionary():
#     global dict_list
#     for s in sensor_list:
#         e_list = []
#         for e in entry_list:
#             if e.SID == s.SID:
#                 e_list.append(e)
#         dict_list[s.SID] = e_list

def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')


def readSensors(filename, conn):
    print("here and now")
    file = open(filename, "r+")
    file.readline()
    while True:
        line = file.readline()
        temp = line.split(",")
        if len(temp) == 7:
            lat = parse_dms(temp[4])
            long = parse_dms(temp[5])
            elev = temp[6].strip()
            if not elev.isdigit():
                elev = -1
            new_sensor = Sensor(temp[0], -1, strip_accents(temp[1]), strip_accents(temp[2]), temp[3], lat, long, elev)
            sensor_list.append(new_sensor)
            # print(new_sensor.toString())
            #insertSensor(conn, new_sensor)
        if not line:
            break
    file.close()


def checkValue(value):
    return value and not value.isspace()


def readData(filename, conn):
    file = open(filename, "r+")
    file.readline()
    global invalid_count
    while True:
        line = file.readline()
        if not line:
            break
        temp = line.split(",")
        entry_date = temp[0]
        pm1_index = 1
        pm25_index = 2
        pm10_index = 3
        temp_index = 4
        for s in sensor_list:
            if checkValue(temp[pm1_index]) and checkValue(temp[pm10_index]) and checkValue(temp[pm25_index]) and checkValue(temp[temp_index]):
                new_entry = Entry(entry_date, int(s.SID), float(temp[pm1_index]), float(temp[pm25_index]),
                                  float(temp[pm10_index]), float(temp[temp_index]))
                insertMeasure(new_entry, conn)
            else:
                invalid_count += 1
            temp_index += 4
            pm1_index += 4
            pm25_index += 4
            pm10_index += 4
    file.close()


def sensorSummary(sensor_id, conn):
    with conn.cursor() as cursor:
<<<<<<< HEAD:edu/pwr/database/DataProcessing.py
        query = 'SELECT * FROM dbo.Measurements WHERE sensorID = %s;'
        data = [sensor_id]
        cursor.execute(query, data)
        data_list = cursor.fetchall()
        # sample code on how to unpack/package the row information from query
        # data_dict = {}
        # for row in data_list:
        #     dk, sid, dt, pm1, pm25, pm10, temp = row
        #     data_dict[dk] = (sid, dt, pm1, pm25, pm10, temp)
=======
        query = 'SELECT date FROM dbo.Measurements WHERE sensorID = %s;'
        query1 = 'SELECT pm1 FROM dbo.Measurements WHERE sensorID = %s;'
        query2 = 'SELECT pm10 FROM dbo.Measurements WHERE sensorID = %s;'
        query3 = 'SELECT pm25 FROM dbo.Measurements WHERE sensorID = %s;'
        query4 = 'SELECT temperature FROM dbo.Measurements WHERE sensorID = %s;'
        data = [sensor_id]

        cursor.execute(query1, data)
        pm1_results = cursor.fetchall()

        cursor.execute(query2, data)
        pm10_results = cursor.fetchall()

        cursor.execute(query3, data)
        pm25_results = cursor.fetchall()

        cursor.execute(query4, data)
        temp_results = cursor.fetchall()

        cursor.execute(query, data)
        date_results = cursor.fetchall()

        f = plt.figure()
        f.set_figwidth(10)
        f.set_figheight(2)

        plt.plot(date_results, pm1_results, label="PM1 Values")
        plt.plot(date_results, pm10_results, label="PM10 Values")
        plt.plot(date_results, pm25_results, label="PM25 Values")
        plt.plot(date_results, temp_results, label="Temperature Values")
        plt.xlabel('Dates')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

>>>>>>> 9ef7d316fdb62ea8147855a9cc36409f4adc2274:edu/pwr/database/main.py
        print("Data for Sensor:", sensor_id)
        print("Total Entries=", len(data_list))
        p = round((len(data_list)/30343)*100, 2)
        print(f'Percentage of Valid Entries={p}%')
        print("")
        conn.commit()


def populateDatabase(conn):
    # createSchema("dbo", conn)
    dropTables(conn)
    print("tables dropped")
    createSensorsTable(conn)
    createMeasurementsTable(conn)
    createTilesTable(conn)
    print("tables created")
    readSensors("C:\\Users\\User\\Desktop\\Multi-Agent\\Sensor_Updates.csv", conn)
    print("sensor table written")
    readData("C:\\Users\\User\\Desktop\\Multi-Agent\\Opole_Historical.csv", conn)
    print("data table written")


def getSensors(conn):
    with conn.cursor() as cursor:
        query1 = 'SELECT sensorID FROM dbo.Sensors;'
        cursor.execute(query1)
        return cursor.fetchall()


def dataSummary(conn):
    sList = getSensors(conn)

    sensorSummary(sList[0], conn)
    #for sensor in sList:
    #   sensorSummary(sensor[0], conn)



