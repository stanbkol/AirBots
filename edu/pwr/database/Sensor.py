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

    def getData(self, conn, start=None, end=None):
        with conn.cursor() as cursor:
            query = 'SELECT * FROM dbo.Measurements WHERE sensorID = %s;'
            data = [self.SID]
            cursor.execute(query, data)
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
            print("Total Entries=", len(data_list))
            p = round((len(data_list) / 30343) * 100, 2)
            print(f'Percentage of Valid Entries={p}%')
            print("")
            conn.commit()