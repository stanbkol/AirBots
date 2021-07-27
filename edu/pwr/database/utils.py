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