import functools
import sqlite3
import time

import pandas as pd
from sqlalchemy import create_engine

from .datenquellen import Datenquelle


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


class SQL(Datenquelle):
    """
    """

    def __init__(self, fname):
        super().__init__(fname)
        self.fname = fname

    def set_connection(self, conn_string=None):
        """

        :param conn_string: zB 'postgresql+psycopg2://scott:tiger@localhost/mydatabase'
        :return:
        """
        self.conn_string = conn_string

    def sqlite(self, tbl_name):
        self.conn = sqlite3.connect(self.fname)
        self.cur = self.conn.cursor()
        return pd.read_sql('SELECT * FROM {}'.format(tbl_name), self.conn)

    def sqlite_exec(self, sql_string):
        try:
            self.cur.execute(sql_string)
        except Exception as e:
            print('[x] FEHLER:', e)

    def posgres(self, tbl_name):
        # TODO: Verbindung herstellen
        engine = create_engine(self.conn_string)
        return pd.read_sql("SELECT * FROM {}".format(tbl_name), engine)


