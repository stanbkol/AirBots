import matplotlib.pyplot as plt

from edu.pwr.database.utils import Field
from edu.pwr.map.MapPoint import calcDistance, MapPoint


class Sensor:
    table_name = "sensors"

    def __init__(self, s_id=None, tid=None, ad1=None, ad2=None, adn=None, lat=None, long=None, elev=None):
        self.sensorid = s_id
        self.tileid = tid
        self.address1 = ad1
        self.address2 = ad2
        self.addressnumber = adn
        self.longitude = long
        self.latitude = lat
        self.elevation = elev
        self.agent = None
        self.state = True

    def convertInterval(self, start_interval, end_interval):
        return ((end_interval-start_interval).days + 1) * 24

    def setAgent(self, a):
        self.agent = a

    def changeState(self, s):
        self.state = s

    def __str__(self):
        return "SID=" + str(self.sensorid) + " Address Line 1:" + str(self.address1) + " Address Line 2:" + str(self.address2) + \
               "Address Number:" + str(self.addressnumber) + " lat: " + str(self.latitude) + " lon: " + \
               str(self.longitude) + " elevation: " + str(self.elevation)

    def metersTo(self, other):
        if isinstance(other, Sensor):
            return calcDistance(startLL=MapPoint(self.latitude, self.longitude),
                                endLL=MapPoint(other.latitude, other.longitude))

    @classmethod
    def get_db_fields(cls, conn):
        with conn.cursor() as cursor:
            query = '''SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s AND table_schema = %s'''
            cursor.execute(query, (cls.table_name, 'dbo'))

            return [Field(name=row[0], data_type=row[1]) for row in cursor.fetchall()]

    def getMeasurements(self, conn, start_interval=None, end_interval=None):
        with conn.cursor() as cursor:
            if not start_interval and not end_interval:
                query = 'SELECT * FROM dbo.Measurements WHERE sensorID = %s;'
                data = [self.sensorid]
            else:
                query = 'SELECT * FROM dbo.Measurements WHERE sensorID = %s AND date BETWEEN %s AND %s;'
                data = [self.sensorid, start_interval, end_interval]
            cursor.execute(query, data)
            cols = []
            for elt in cursor.description:
                cols.append(elt[0])
            data_list = cursor.fetchall()

            # sample code on how to unpack/package the row information from query
            # data_dict = {}
            # for row in data_list:
            #     dk, sid, dt, pm1, pm25, pm10, temp = row
            #     data_dict[dk] = (sid, dt, pm1, pm25, pm10, temp)

            # sample code for basic numpy scatterplots
            # f = plt.figure()
            # f.set_figwidth(10)
            # f.set_figheight(2)
            # plt.plot(date_results, pm1_results, label="PM1 Values")
            # plt.plot(date_results, pm10_results, label="PM10 Values")
            # plt.plot(date_results, pm25_results, label="PM25 Values")
            # plt.plot(date_results, temp_results, label="Temperature Values")
            # plt.xlabel('Dates')
            # plt.ylabel('Values')
            # plt.legend()
            # plt.show()

            print("Data for Sensor:", self.sensorid)
            print("Valid Entries=", len(data_list))
            print("Total Entries=", self.convertInterval(start_interval, end_interval))
            p = round((len(data_list) / self.convertInterval(start_interval, end_interval)) * 100, 2)
            print(f'Percentage of Valid Entries={p}%')
            print("")
            conn.commit()

            return cols, data_list

    @classmethod
    def sensor_set_fields(cls, **row_data):
        sensor = cls()
        for field_name, value in row_data.items():
            setattr(sensor, field_name, value)
        return sensor