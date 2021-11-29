import math
from src.map.HexGrid import DwHex, dw_to_hex, hex_distance, dw_distance, hex_linedraw, hex_to_dw, Hex, \
    tile_dist_trust_factors


def hex_path_test():
    dwa = DwHex(0, 0)
    dwb = DwHex(0, 4)
    a = dw_to_hex(dwa)
    b = dw_to_hex(dwb)
    print(f'hex dist: %s' % hex_distance(a, b))
    print(f'dw dist: %s' % dw_distance(dwa, dwb))
    print(f'From %s to %s:' % (str(a), str(b)))
    # cb = cb.add_cube(epsilon)
    path = hex_linedraw(a, b)
    for h in path:
        dw = hex_to_dw(h)
        print(str(h), str(dw))


def genHexRect_Test():
    y_max = 3
    x_max = 4
    x_min = 0
    tiles = []
    print("")
    for r in range(0, y_max+1, 1):
        r_offset = math.floor(r / 2.0)
        for q in range(x_min-r_offset, (x_max-r_offset)+1, 1):
            h = Hex(q,r,-q-r)
            tiles.append(h)

    print("dw rep:")
    for h in tiles:
        print(f'Hex: %s,\t\tDw: %s' % (str(h), str(dw_to_hex(hex_to_dw(h)))))


def tile_tcf_test():
    tfs = tile_dist_trust_factors()
    print(tfs)
    print(f"len: {len(tfs)}")
    dist = 800
    closest = min(range(1, 7 + 1), key=lambda x: abs(x - dist))
    print(closest)
    print(f"dist factor at {dist}: {tfs[closest - 1]}")


if __name__ == '__main__':
    hex_path_test()