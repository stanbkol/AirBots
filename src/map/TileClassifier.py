
def classifyT(tile_data):
    if len(tile_data["buildings"]) == 0:
        return "Empty Tile Error"
    else:
        L1 = classifyL1(tile_data["buildings"])
        L2 = classifyL2(tile_data["roads"])
        return L2+" "+L1


def classifyL1(building_data):
    building_dict = {}
    for b in building_data:
        # print(b)
        if b[0] in building_dict.keys():
            building_dict[b[0]] += b[1]
        else:
            building_dict[b[0]] = b[1]
    # print(building_dict)
    return max(building_dict, key=building_dict.get)


def classifyL2(road_data):
    road_dict = {}
    for r in road_data:
        if r[0] in road_dict.keys():
            road_dict[r[0]] += 1
        else:
            road_dict[r[0]] = 1
    #print(road_dict)
    if "motorway" in road_dict.keys() or "trunk" in road_dict.keys():
        return "T1"
    if "primary" in road_dict.keys() or "secondary" in road_dict.keys():
        return "T2"
    else:
        return "T3"