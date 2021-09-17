from datetime import datetime
from edu.pwr.airbots.wma import show_wma
from edu.pwr.database.DataLoader import createConnection


def main():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 8, 0)

    #conn = createConnection()
    #show_wma(conn, start, end)


if __name__ == '__main__':
    main()
