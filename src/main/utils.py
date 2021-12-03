import decimal
import re
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def p_err(a, p):
    num_preds = len(a)
    total = 0
    for i in range(0, num_preds):
        temp = abs((p[i] - a[i]) / a[i])
        total += temp
    return round((total / num_preds) * 100, 2)


def MSE(a, p):
    actual, pred = np.array(a), np.array(p)
    return round(mean_squared_error(actual, pred), 2)


def MAE(a, p):
    actual, pred = np.array(a), np.array(p)
    return round(mean_absolute_error(actual, pred), 2)


def countInterval(start, end):
    diff = end - start
    days, seconds = diff.days, diff.seconds
    total_intervals = days * 24 + seconds // 3600
    return total_intervals+1


def unique_tiles(tiles):
    unique_tiles = {}

    for t in tiles:
        if t.tid not in unique_tiles.keys():
            unique_tiles[t.tid] = t

    return [unique_tiles[k] for k, v in unique_tiles.items()]


def tile_ranges(sensor_tiles, r=3):
    """
    given list of tiles and range, returns a list of unique tiles that are in range of each
    tile provided.
    """
    tiles = []
    for t in sensor_tiles:
        neighbors = t.tiles_in_range(r)  # this includes the center tile: t
        tiles.extend(neighbors)
        print(f"\tgot {len(neighbors)} neighbors for tile {t.tid}")

    return unique_tiles(tiles)

def getJson(file):
    f = open(file, encoding="utf8")
    data = json.load(f)
    return data

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