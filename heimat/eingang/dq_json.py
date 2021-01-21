import functools
import time

import pandas as pd
import requests

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


class JSON(Datenquelle):

    def __init__(self, fname):
        super().__init__(fname)
        self.fname = fname

    def lesen(self):
        self.data = pd.read_json(self.fname, encoding="utf-8")

    def api(self, url):
        """
            Aufruf API-URLs
        :param url:
            z.B. http://api.worldbank.org/v2/countries/br;cn;us;de/indicators/SP.POP.TOTL/?format=json&per_page=1000
        :return:
        """
        r = requests.get(url)
        res = r.json()
        if len(res) != 0:
            return pd.DataFrame(res[1])
        else:
            return None

    def __add__(self, csv_obj):
        return pd.concat([self.data, csv_obj.data])

    def __repr__(self):
        return self.data.head(5)
