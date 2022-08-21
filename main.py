import os.path

import tensorflow as tf
from tensorflow import keras

from modules.dataset import get_np_dataset
from modules import tf_nn


def main():
    mean, diff = 2048, 512
    train_points, test_points, train_labels, test_labels, class_map = get_np_dataset(mean=mean, diff=diff)

    # np.ndarray からデータセット (generator) 生成
    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

    # データ順のシャッフル・(ノイズ追加・点のシャッフル)・バッチ化
    batch_size = 32
    train_dataset = train_dataset.shuffle(len(train_points)).batch(batch_size)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(batch_size)

    num_classes = 10
    if os.path.exists("./output/point_net"):
        models = keras.models.load_model()
    else:
        models = tf_nn.point_net(mean + int(diff/2), num_classes)
        tf_nn.fit_tf_model(models, train_dataset, test_dataset)


if __name__ == "__main__":
    main()