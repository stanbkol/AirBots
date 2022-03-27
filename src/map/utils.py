from src.database.Models import getTilesORM

def targetInterval(time, interval):
    return time - timedelta(hours=interval), time


def tileSummary():
    tiles = getTilesORM(1)
    tc = {}
    for t in tiles:
        if t.tclass in tc.keys():
            tc[t.tclass] += 1
        else:
            tc[t.tclass] = 1
    return tc

def archiveResults(docs):
    filenames = next(walk(docs), (None, None, []))[2]
    for file in filenames:
        temp = file.split('_')
        if temp[-1] == 'results.xlsx':
            old_file_path = docs+"\\"+file
            new_file_path = docs+"\\results\\"+file
            os.replace(old_file_path, new_file_path)