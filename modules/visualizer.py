from tensorboard import program


def open_tensorboard(log_path):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_path])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

