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