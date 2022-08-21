import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'{name}: {time.time() - t0:.3f} s')
