import glob
import os.path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from modules.dataset import get_np_dataset
from modules import nn, visualizer
import matplotlib.pyplot as plt


def plot_samples(points, predictions, labels, class_map):
    # plot points with predicted class and label
    fig = plt.figure(figsize=(15, 10))
    points = points.numpy()
    for i in range(8):
        predicted_label = class_map[str(predictions[i].numpy())]
        true_label = class_map[str(labels.numpy()[i])]

        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
        ax.set_title(f"pred: {predicted_label}, label: {true_label}")
        ax.set_axis_off()
    plt.show()


def get_accuracy(model, test_dataset, class_map, draw):
    data_num, true_num = 0, 0
    for dataset_idx in range(1, len(test_dataset) + 1):
        # パッチサイズ分だけ取得
        data = test_dataset.take(dataset_idx)
        points, labels = list(data)[0]

        # 各ラベルに対する予測確率を取得
        predictions = model.predict(points)
        # 確率が最も高いラベルを取得
        predictions = tf.math.argmax(predictions, -1)

        data_num += len(labels)
        true_num += np.count_nonzero(predictions.numpy() == labels.numpy())
        if draw:
            plot_samples(points, predictions, labels, class_map)
    print(f"Prediction accuracy: {true_num / data_num}")


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
        model = keras.models.load_model("./output/point_net")
    else:
        model = nn.point_net(mean + int(diff / 2), num_classes)
        nn.fit_tf_model(model, train_dataset, test_dataset)

    log_file_list = glob.glob("./output/fit/*")
    visualizer.open_tensorboard(log_file_list[-1])

    get_accuracy(model, test_dataset, class_map, draw=False)


if __name__ == "__main__":
    main()
