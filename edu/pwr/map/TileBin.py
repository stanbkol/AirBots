from edu.pwr.map.MapPoint import calcCoordinate, calcDistance
import decimal
from edu.pwr.database.utils import drange


class TileBin:
    sensor_bin = []

    def __init__(self, mapID, tileID, center, numSides, coordinates=None, max_elevation=None, min_elevation=None,
                 diameter=100, tileType=None, temp=None, pm1=None, pm10=None, pm25=None, polyString=""):
        # List of MapPoints
        if coordinates is None:
            coordinates = []
        # if not any(isinstance(c, MapPoint) for c in coordinates):
        #     raise TypeError("coordinates must be of type MapPoint")
        self.tileId = tileID
        # map it belongs to
        self.mapId = mapID
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
        self.poly_str = polyString

    def getSensors(self):
        return self.sensor_bin

    def getNeighborTiles(self):
        # DB query, in what order? Northeastern-most going clockwise?
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

    def set_PolyString(self, poly):
        if isinstance(poly, str):
            self.poly_str = poly
            return True

        return False

    def set_vertices(self, vertex_list):
        if len(vertex_list) == self.numSides:
            for i in self.numSides:
                self.coordinates.append(vertex_list[i])

    def metersTo(self, other):
        if isinstance(other, TileBin):
            return calcDistance(startLL=self.centerPt, endLL=other.centerPt)


if __name__ == '__main__':
    pass
