from edu.pwr.database.utils import Field


class Entry:
    table_name = "measurements"

    def __init__(self, date_key=None, sensor_id=None, date=None, pm1=None, pm25=None, pm10=None, temperature=None):
        self.date = date
        self.dk = date_key
        self.sid = sensor_id
        self.temp = temperature
        self.pm1 = pm1
        self.pm10 = pm10
        self.pm25 = pm25

    def __str__(self):
        return "D="+self.date+" SID="+self.SID + " Temp="+self.temp + " Pm25="+self.pm25 + " Pm1="+self.pm1 + " Pm10="+self.pm10

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