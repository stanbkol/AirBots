from edu.pwr.map.MapPoint import MapPoint, calcCoordinate
import decimal


class TileBin:
    sensor_bin = []

    def __init__(self, mapID, tileID, center, numSides, coordinates=None, max_elevation=None, min_elevation=None,
                 diameter=100, tileType=None, temp=None, pm1=None, pm10=None, pm25=None):
        # List of MapPoints
        if coordinates is None:
            coordinates = []
        # if not any(isinstance(c, MapPoint) for c in coordinates):
        #     raise TypeError("coordinates must be of type MapPoint")
        self.tid = tileID
        # map it belongs to
        self.map = mapID
        self.coordinates = coordinates
        self.diameter = diameter
        self.centerPt = center
        self.tileClass = tileType
        self.max_elevation = max_elevation
        self.min_elevation = min_elevation
        self.temperature = temp
        self.pm10_avg = pm10
        self.pm1_avg = pm1
        self.pm25_avg = pm25
        self.numSides = numSides

    def getSensors(self):
        return self.sensor_bin

    def getNeighborTiles(self):
        # DB query, in what order? Northernmost going clockwise?
        pass

    def setClass(self, tileClass):
        self.tileClass = tileClass

    def setSensors(self, sensors):
        self.sensor_bin.append(sensors)

    def generate_vertices_coordinates(self):
        # vertices = []
        radius = self.diameter / 2
        degs = list(drange(0, 360, 360 / self.numSides))
        for d in degs:
            vertex_coor = calcCoordinate(self.centerPt, radius, d)
            # vertices.append(vertex_coor)
            self.coordinates.append(vertex_coor)

        return self.coordinates


def drange(start, stop, jump):
    while start < stop:
        yield float(start)
        start += decimal.Decimal(jump)


if __name__ == '__main__':
    print(list(drange(0, 360, 360 / 6)))
    pass
