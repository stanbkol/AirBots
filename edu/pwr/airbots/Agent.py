
def getData(sid, start_date, end_date):
    pass


# switcher method where we will input the algorithms we settle on
def algChoice(choice):
    switcher = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }
    return switcher.get(choice, "Invalid Option")


class Agent:
    def __init__(self, s, c):
        self.sensor = s
        self.alg = algChoice(c)

    def updateAlg(self, c):
        self.alg = algChoice(c)

    def makePrediction(self, sensor):
        getData(sensor.sid, "derp", "lerp")
