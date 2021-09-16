import matplotlib.pyplot as plt

from edu.pwr.database.utils import Field
from edu.pwr.map.MapPoint import calcDistance, MapPoint


class Sensor:
    table_name = "sensors"

    def __init__(self, sensor_id=None, tile_id=None, address1=None, address2=None, address_num=None, latitude=None,
                 longitude=None, elevation=None):
        self.sensorid = sensor_id
        self.tileid = tile_id
        self.address1 = address1
        self.address2 = address2
        self.addressnumber = address_num
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.agent = None
        self.state = True

    def convertInterval(self, start_interval, end_interval):
        return ((end_interval-start_interval).days + 1) * 24

    def setAgent(self, a):
        self.agent = a

    def changeState(self, s):
        self.state = s

    def __str__(self):
        return "sensorid=" + str(self.sensorid) + " tileid=" + str(self.tileid) + " Address Line 1:" + str(self.address1) + " Address Line 2:" + str(self.address2) + \
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

            return cols, data_list

    @classmethod
    def sensor_set_fields(cls, **row_data):
        sensor = cls()
        for field_name, value in row_data.items():
            setattr(sensor, field_name, value)
        return sensor