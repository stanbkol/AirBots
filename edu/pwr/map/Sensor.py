class Sensor:
    def __init__(self, s_id, ad1, ad2, adn, long, lat, elev):
        self.SID = s_id
        self.address_1 = ad1
        self.address_2 = ad2
        self.address_num = adn
        self.longitude = long
        self.latitude = lat
        self.elevation = elev
        self.agent = None
        self.state = True

    # set the sensor to a type of agent (algorithm)
    def setAgent(self, a):
        self.agent = a

    # find the closest sensor
    def findNearest(self):
        pass

    def changeState(self, s):
        self.state = s

    def toString(self):
        print(
            "SID=" + self.SID + " Address Line 1:" + self.address_1 + " Address Line 2:" + self.address_2 + " Address Number:" + self.address_num)
