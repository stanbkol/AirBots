import unicodedata
from datetime import datetime
import psycopg2
from edu.pwr.database.utils import *
from edu.pwr.database.Entry import *
from edu.pwr.database.Sensor import *
from edu.pwr.database.DataLoader import *
from shapely.geometry import Polygon, Point

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
            # insertSensor(conn, new_sensor)
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
            if checkValue(temp[pm1_index]) and checkValue(temp[pm10_index]) and checkValue(
                    temp[pm25_index]) and checkValue(temp[temp_index]):
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


def dataSummary(conn):
    sList = getSensors(conn)
    for sensor in sList:
        sensorSummary(sensor[0], conn)


def updateSensorsTile(conn):
    sensor_lon = 6
    sensor_lat = 5
    sid = 0
    tid = 0
    # tiles 5-10 for vertices la,lo
    sensors = getSensors(conn)
    tiles = getTiles(conn)
    for sensor in sensors:
        sensor_marker = Point(sensor[sensor_lon], sensor[sensor_lat])
        sensor_id = sensor[sid]
        for tile in tiles:
            vertices = []
            for vertex_col in range(5, 11):
                lonlat = tuple([float(c) for c in str(tile[vertex_col]).split(",")][::-1])
                vertices.append(lonlat)

            tile_poly = Polygon(vertices)
            if tile_poly.contains(sensor_marker):
                update_sensor_tile(conn, sensor_id, tile[tid])
