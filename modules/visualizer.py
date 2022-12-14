from tensorboard import program

from modules import utils


def open_tensorboard(log_path):
    utils.logger.info(f"Open for Tensorboard: {log_path}")

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_path])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
