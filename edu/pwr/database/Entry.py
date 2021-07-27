class Entry:
    def __init__(self, d, s_id, pm1, pm25, pm10, t):
        self.date = d
        self.SID = s_id
        self.temp = t
        self.pm1 = pm1
        self.pm10 = pm10
        self.pm25 = pm25

    def __str__(self):
        return "D="+self.date+" SID="+self.SID + " Temp="+self.temp + " Pm25="+self.pm25 + " Pm1="+self.pm1 + " Pm10="+self.pm10
