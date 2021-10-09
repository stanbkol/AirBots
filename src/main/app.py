from src.map.Model import *
from src.database.Models import *
from src.database.DbManager import engine


def main():
    # Model(r"C:\Users\mrusieck\PycharmProjects\AirBot\Docs\Model1")
    # classifyTiles(r"C:\Users\User\Desktop\Multi-Agent\tile_info")
    createAllTables(engine)
    populateTables()


if __name__ == '__main__':
    main()
