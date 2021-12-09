from datetime import datetime

from src.map.Central import *
from src.database.Models import *


def main():
    c = Central('..\\..\\..\\AirBots\\docs\\Model1')
    c.makePrediction(5697, datetime(2020, 1, 7, 0))


if __name__ == '__main__':
    main()
