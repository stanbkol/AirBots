from edu.pwr.database.DataProcessing import *
import psycopg2


def main():
    print("connecting to server")
    conn = psycopg2.connect(
        host="pgsql13.asds.nazwa.pl",
        database="asds_PWR",
        user="asds_PWR",
        password="W#4bvgBxDi$v6zB")
    dataSummary(conn)
    conn.close()


if __name__ == '__main__':
    main()
