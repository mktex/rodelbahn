import functools
import time

from bs4 import BeautifulSoup

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


class XML(Datenquelle):
    """
    """

    def __init__(self, fname):
        super().__init__(fname)
        self.fname = fname

    def lesen(self):
        with open(self.fname) as fp:
            soup = BeautifulSoup(fp, "lxml")
        i = 0
        for record in soup.find_all('record'):
            i += 1
            for rec in record.find_all('field'):
                print(rec['name'], ': ', rec.text)
            print()
            if i == 5:
                break

