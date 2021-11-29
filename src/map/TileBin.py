from src.map.MapPoint import calcCoordinate, calcDistance, MapPoint
from src.main.utils import drange, Field


class TileBin:
    sensor_bin = []
    table_name = "tiles"

    def __init__(self, tileID=None, mapID=None, center=None, numSides=None, coordinates=None, max_elevation=None, min_elevation=None,
                 diameter=100, tileType=None, temp=None, pm1=None, pm10=None, pm25=None, polyString=""):
        # List of MapPoints
        if coordinates is None:
            coordinates = []
        # if not any(isinstance(c, MapPoint) for c in coordinates):
        #     raise TypeError("coordinates must be of type MapPoint")
        self.tileid = tileID
        # map it belongs to
        self.mapid = mapID
        self.coordinates = coordinates
        self.diameter = diameter
        self.centerlatlon = center
        self.tclass = tileType
        self.max_elevation = max_elevation
        self.min_elevation = min_elevation
        self.temperature = temp
        self.pm10_avg = pm10
        self.pm1_avg = pm1
        self.pm25_avg = pm25
        self.numSides = numSides
        self.poly_str = polyString

    def __repr__(self):
        return "<TileBin(tileid='%s',mapid='%s', center='%s', sides='%s', type='%s')>" % (self.tileid, self.mapid, self.centerlatlon, self.numSides,
                                                                           self.tclass)

    @classmethod
    def tilebin_set_fields(cls, **row_data):
        tilebin = cls()

        vertices = []
        tclass = dict()

        # popping special-case fields for processing
        for k, d in row_data.items():
            if k.startswith('vertex'):
                v = row_data[k]
                vertices.append(db_strmp(v))
            if k.startswith('class'):
                tclass[k] = row_data[k]

        discard = ['vertex', 'class']
        row_data = {k:v for k,v in row_data.items() if not any(d in k for d in discard)}

        setattr(tilebin, 'numSides', len(vertices))
        tilebin.set_vertices(vertices)
        setattr(tilebin, list(tclass.keys())[0], list(tclass.values())[0])

        for field_name, value in row_data.items():
            setattr(tilebin, field_name, value)
        return tilebin

    @classmethod
    def get_db_fields(cls, conn):
        with conn.cursor() as cursor:
            query = '''SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s AND table_schema = %s'''
            cursor.execute(query, (cls.table_name, 'dbo'))

            return [Field(name=row[0], data_type=row[1]) for row in cursor.fetchall()]

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
            vertex_coor = calcCoordinate(self.centerlatlon, radius, d)
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
            for i in range(self.numSides):
                self.coordinates.append(vertex_list[i])

    def metersTo(self, other):
        if isinstance(other, TileBin):
            return calcDistance(startLL=self.centerlatlon, endLL=other.centerlatlon)


def db_strmp(db_str):
    if "," in db_str:
        vertices = tuple([float(c) for c in str(db_str).split(",")])
        return MapPoint(vertices[0], vertices[1])
    return db_str

if __name__ == '__main__':
    pass
