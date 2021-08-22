from edu.pwr.database.DataProcessing import *
import psycopg2


def main():
    print("connecting to server")
    conn = psycopg2.connect(
        host="pgsql13.asds.nazwa.pl",
        database="asds_PWR",
        user="asds_PWR",
        password="W#4bvgBxDi$v6zB")

    start = datetime(2019, 12, 1, 0)
    end = datetime(2019, 12, 31, 23)
    dataSummary(conn, start, end)
    conn.close()


if __name__ == '__main__':
    main()
