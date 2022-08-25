import glob
import os.path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from modules.dataset import get_np_dataset
from modules import nn, visualizer
import open3d as o3d
import matplotlib.pyplot as plt


def get_accuracy(model, test_dataset, class_map, draw):
    data_num, true_num = 0, 0
    for dataset_idx in range(1, len(test_dataset) + 1):
        # パッチサイズ分だけ取得
        data = test_dataset.take(dataset_idx)
        point_clouds, labels = list(data)[0]

        # 各ラベルに対する予測確率を取得
        predictions = model.predict(point_clouds)
        # 確率が最も高いラベルを取得
        predictions = tf.math.argmax(predictions, -1)
        is_correct_list = predictions.numpy() == labels.numpy()
        false_indexes = np.arange(0, len(labels))[~is_correct_list]

        if draw:
            for false_index in false_indexes:
                points = np.array(point_clouds[false_index])
                points = points[points[:, 3] == 1][:, :3]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                predict_label = class_map[str(predictions[false_index].numpy())]
                true_label = class_map[str(labels.numpy()[false_index])]
                window_name = f"prediction: {predict_label}, label: {true_label}"
                o3d.visualization.draw_geometries([pcd], window_name=window_name)

        data_num += len(labels)
        true_num += np.count_nonzero(is_correct_list)

    accuracy = true_num / data_num
    print(f"Prediction accuracy: {accuracy}")


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
