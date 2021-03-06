import re
from geopy import distance
from numpy import arctan2, sin, cos
import math
from geopy.distance import Distance


class MapPoint:
    def __init__(self, latitude, longitude, name='Unknown'):
        self.lat = latitude
        self.lon = longitude
        self.name = name

    def __eq__(self, other):
        if isinstance(other, MapPoint):
            return other.lat == self.lat and other.lon == self.lon
        return False

    @property
    def latitude(self):
        return self.lat

    @property
    def longitude(self):
        return self.lon

    @property
    def LatLonCoords(self):
        return (self.lat, self.lon)

    @property
    def LonLatCoords(self):
        return (self.lon, self.lat)

    @property
    def latlon_str(self):
        return str(self.lat) + "," + str(self.lon)

    @classmethod
    def createFromStr(cls, latlon_str):
        c_strings = latlon_str.split(",")
        return MapPoint(latitude=float(c_strings[0]), longitude=float(c_strings[1]))

    def __str__(self):
        return self.name + '_' + '(' + str(self.lat) + ',' + str(self.lon) + ')'

    def dms2dd(self, degrees, minutes, seconds, direction):
        dd = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60)
        if direction == 'E' or direction == 'S':
            dd *= -1
        return dd

    def parse_dms(self, dms):
        # parts = re.split('[^\d\w]+', dms)
        parts = re.split(' ', dms)
        if len(parts) < 4:
            return -1
        return self.dms2dd(parts[0], parts[1], parts[2], parts[3])

    def setName(self, name):
        self.name = name

    def bearing(self, destination):
        dL = destination.longitude - self.longitude
        X = cos(destination.latitude) * sin(dL)
        Y = cos(self.latitude) * sin(destination.latitude) - sin(self.latitude) * cos(destination.latitude) * cos(dL)
        # in radians
        bearing = arctan2(X, Y)
        return math.degrees(bearing)


# degrees: 0 – North, 90 – East, 180 – South, 270 or -90 – West.
def calcCoordinate(startLL, dist, degrees):
    coors = distance.distance(meters=dist).destination(startLL.LatLonCoords, bearing=degrees, distance=Distance(meters=dist))
    return MapPoint(round(coors.latitude, 8), round(coors.longitude, 8))


def calcDistance(startLL, endLL):
    if isinstance(startLL, MapPoint) and isinstance(endLL, MapPoint):
        return distance.distance(startLL.LatLonCoords, endLL.LatLonCoords).meters
    return distance.distance(startLL, endLL).meters


directions = {"north": 0, "east": 90, "south": 180, "west": 270}
