from datetime import datetime
import time
from contextlib import contextmanager
from logging import Formatter, StreamHandler, FileHandler, getLogger, DEBUG


_formatter = Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s")
_stream_handler = StreamHandler()
_stream_handler.setLevel(DEBUG)
_stream_handler.setFormatter(_formatter)
_log_file_name = datetime.now().strftime("%Y%m%d_%H%M%S")
_file_handler = FileHandler(f"./logs/{_log_file_name}.log")
_file_handler.setFormatter(_formatter)

logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.addHandler(_stream_handler)
logger.propagate = False


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'{name}: {time.time() - t0:.3f} s')
