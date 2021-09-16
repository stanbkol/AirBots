import decimal
import re


def dms2dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);
    if direction == 'E' or direction == 'S':
        dd *= -1
    return dd


def parse_dms(dms):
    # parts = re.split('[^\d\w]+', dms)
    parts = re.split(' ', dms)
    if len(parts) < 4:
        return -1
    return dms2dd(parts[0], parts[1], parts[2], parts[3])


def drange(start, stop, jump):
    while start < stop:
        yield float(start)
        start += decimal.Decimal(jump)


class Field:
    def __init__(self, name, data_type):
        self.name = name
        self.data_type = data_type

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name} ({self.data_type})>"