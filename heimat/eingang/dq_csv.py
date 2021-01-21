import functools
import time

import pandas as pd

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


class CSV(Datenquelle):
    """
    """
    _args = None
    _kwargs = None

    def __init__(self, fname, *args, **kwargs):
        super().__init__(fname)
        self._args = args
        self._kwargs = kwargs

    def lesen(self):
        self.data = pd.read_csv(self.fname, *self._args, **self._kwargs)
        self.__repr__()

    def mapping(self, xkateg1, xkateg2):
        """
            Hat man zwei kategoriale Variablen kateg1 und kateg2, so kann man einen Dict-Objekt generieren
        :param xkateg1:
        :param xkateg2:
        :return:
        """
        list_dicts = self.data[[xkateg1, xkateg2]].drop_duplicates().transpose().to_dict().values()
        mapping_res = {}
        for xdict in list_dicts:
            mapping_res[xdict[xkateg1]] = xdict[xkateg2]
        return mapping_res

    def __add__(self, csv_obj):
        """
        """
        return pd.concat([self.data, csv_obj.data])

    def __repr__(self):
        """
        """
        if self.data is None:
            self.lesen()
        print(self.data.head(3))
        print("...")
        print(self.data.tail(3))
        print("[x] Datenbestand geladen:", self.data.shape)
        return ''
