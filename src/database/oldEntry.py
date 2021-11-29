from src.main.utils import Field


class Entry:
    table_name = "measurements"

    def __init__(self, date_key=None, sensor_id=None, date=None, pm1=None, pm25=None, pm10=None, temperature=None):
        self.datekey = date_key
        self.sensorid = sensor_id
        self.date = date
        self.temperature = temperature
        self.pm1 = pm1
        self.pm10 = pm10
        self.pm25 = pm25

    def __str__(self):
        return "Measure: date=" + str(self.date) +" SID=" + str(self.sensorid) + " Temp=" + str(self.temperature) + " Pm25=" + str(self.pm25) + " Pm1=" + str(self.pm1) + " Pm10=" + str(self.pm10)

    def __repr__(self):
        return "<Measure(datekey='%s',date='%s', sensorid='%s', temp='%s', pm25='%s', pm10='%s', pm1='%s')>" % (
            self.datekey, self.date, self.sensorid, self.temperature, self.pm25, self.pm10, self.pm1)

    @classmethod
    def get_db_fields(cls, conn):
        with conn.cursor() as cursor:
            query = '''SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s AND table_schema = %s'''
            cursor.execute(query, (cls.table_name, 'dbo'))
            return [Field(name=row[0], data_type=row[1]) for row in cursor.fetchall()]

    @classmethod
    def entry_set_fields(cls, **row_data):
        entry = cls()
        for field_name, value in row_data.items():
            setattr(entry, field_name, value)
        return entry