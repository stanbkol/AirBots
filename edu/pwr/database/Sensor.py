import matplotlib.pyplot as plt


class Sensor:
    def __init__(self, s_id, tid, ad1, ad2, adn, lat, long, elf):
        self.SID = s_id
        self.tile = tid
        self.address_1 = ad1
        self.address_2 = ad2
        self.address_num = adn
        self.latitude = lat
        self.longitude = long
        self.elevation = elf

    def toString(self):
        print("SID=" + self.SID + " Address Line 1:" + self.address_1 + " Address Line 2:" + self.address_2 + " Address Number:" + self.address_num)

    def convertInterval(self, start_interval, end_interval):
        return ((end_interval-start_interval).days + 1) * 24

    def getData(self, conn, start_interval=None, end_interval=None):
        with conn.cursor() as cursor:
            if not start_interval and not end_interval:
                query = 'SELECT * FROM dbo.Measurements WHERE sensorID = %s;'
                data = [self.SID]
            else:
                query = 'SELECT * FROM dbo.Measurements WHERE sensorID = %s AND date BETWEEN %s AND %s;'
                data = [self.SID, start_interval, end_interval]
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

            print("Data for Sensor:", self.SID)
            print("Valid Entries=", len(data_list))
            print("Total Entries=", self.convertInterval(start_interval, end_interval))
            p = round((len(data_list) / self.convertInterval(start_interval, end_interval)) * 100, 2)
            print(f'Percentage of Valid Entries={p}%')
            print("")
            conn.commit()

            return cols, data_list
