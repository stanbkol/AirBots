# weights based on distance for calculating moving average
from src.map.MapPoint import calcDistance, MapPoint

A = MapPoint(1,1)
B = MapPoint(2,2)

def main():
    print(calcDistance(A,B))

if __name__ == '__main__':
    main()