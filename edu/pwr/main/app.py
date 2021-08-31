from edu.pwr.database.DataProcessing import *
import psycopg2


def main():
    global invalid_count
    print("connecting to server")
    conn = createConnection()
    dropTiles(conn)
    createTilesTable(conn)
    conn.close()


if __name__ == '__main__':
    main()
