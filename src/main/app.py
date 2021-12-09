from datetime import datetime

from src.map.Central import *
from src.database.Models import *


def main():
    c = Central('..\\..\\..\\AirBots\\docs\\Model2')
    c.makePrediction(5697, datetime(2020, 10, 24, 0))


if __name__ == '__main__':
    main()
