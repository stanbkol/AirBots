from datetime import datetime
from src.map.Central import *


def main():
    # archiveResults('..\\..\\..\\AirBots\\docs')
    model = Central('..\\..\\..\\AirBots\\docs\\Model2')
    model.makePrediction(5697, datetime(2020, 9, 15, 0))


if __name__ == '__main__':
    main()
