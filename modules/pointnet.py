import datetime
import os
import glob
import os.path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from modules.dataset import get_np_dataset
from modules import visualizer
import open3d as o3d


def conv_bn(shape, filters):
    """
    バッチ正則化付き pointwise 畳込み層
    """
    inputs = keras.Input(shape=shape)
    x = keras.layers.Conv1D(filters, kernel_size=1, padding="valid")(inputs)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    outputs = keras.layers.Activation("relu")(x)
    return keras.Model(inputs, outputs, name="conv_bn")


def dense_bn(shape, filters):
    """
    バッチ正則化付き全結合層
    """
    inputs = keras.Input(shape=shape)
    x = keras.layers.Dense(filters)(inputs)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    outputs = keras.layers.Activation("relu")(x)
    return keras.Model(inputs, outputs, name="dense_bn")


def orthogonal_regularizer(x, num_features):
    l2reg = 0.001

    # (バッチ数, 空間次元, 空間次元) の形に変形
    x = tf.reshape(x, (-1, num_features, num_features))
    # 3成分目同士で内積 → 成分は (バッチ数, 空間次元, バッチ数, 空間次元)
    xxt = tf.tensordot(x, x, axes=(2, 2))
    # (バッチ数, 空間次元, 空間次元) の形に変形
    xxt = tf.reshape(xxt, (-1, num_features, num_features))
    # reduce_sum は次元の指定がなければ全成分の総和
    return tf.reduce_sum(l2reg * tf.square(xxt - tf.eye(num_features)))


def tnet(shape, num_features):
    """
    T-net
    :param shape: 入力データの形状
    :param num_features: 回転する次元数
    :return:
    """
    inputs = keras.Input(shape=shape)

    # 近似的な対称関数を適用 (変換してから対称化)
    x = conv_bn(shape, 32)(inputs)
    x = conv_bn(shape, 64)(x)
    x = conv_bn(shape, 512)(x)
    x = keras.layers.GlobalMaxPooling1D(x)

    # 直交行列を学習
    x = dense_bn(shape, 256)(x)
    x = dense_bn(shape, 128)(x)
    x = keras.layers.Dense(
        num_features * num_features,  # 直交行列の成分数
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=lambda mat: orthogonal_regularizer(mat, num_features),
    )(x)
    feature_mat = keras.layers.Reshape((num_features, num_features))(x)

    # Apply affine transformation to input features
    outputs = keras.layers.Dot(axes=(2, 1))([x, feature_mat])
    return keras.Model(inputs, outputs, name="dense_bn")


def point_net(point_num, class_num):
    """
    Functional API で PointNet を定義.
    (cf. https://dev.classmethod.jp/articles/tensorflow-keras-api-pattern/)

    :param point_num:
    :param class_num:
    :return:
    """
    encoder_inputs = keras.Input(shape=(point_num, 4))
    # 3次元空間の正準座標を求めて適用
    x = tnet((point_num, 4), 3)(encoder_inputs)
    # 特徴量抽出
    x = conv_bn((point_num, 4), 32)(x)
    x = conv_bn((point_num, 32), 32)(x)
    # 特徴量空間の正準座標を求めて適用
    x = tnet((point_num, 32), 32)(x)
    # 近似的な対称関数を適用 (混合してから対称化)
    x = conv_bn((point_num, 32), 32)(x)
    x = conv_bn((point_num, 32), 64)(x)
    x = conv_bn((point_num, 64), 512)(x)
    encoder_outputs = keras.layers.GlobalMaxPooling1D(x)
    encoder_model = keras.Model(encoder_inputs, encoder_outputs, name="pointnet_encoder")

    # クラス分類
    classifier_inputs = keras.Input(shape=(512, ))
    x = dense_bn((512,), 256)(classifier_inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = dense_bn((256,), 128)(x)
    x = keras.layers.Dropout(0.3)(x)
    # 確率出力
    classifier_outputs = keras.layers.Dense(class_num, activation="softmax")(x)
    classifier_model = keras.Model(classifier_inputs, classifier_outputs, name="pointnet_classifier")

    inputs = keras.Input(shape=(point_num, 4))
    model = keras.Model(inputs, classifier_model(encoder_model(x)), name="point_net")
    model.summary()

    return model


def fit_tf_model(model, train_dataset, test_dataset):
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )

    log_dir = "./output/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(train_dataset, epochs=20, validation_data=test_dataset,
              callbacks=[tensorboard_callback])
    model.save("./output/point_net")


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
    mean, diff = 2048, 0
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
        model = point_net(mean + int(diff / 2), num_classes)
        fit_tf_model(model, train_dataset, test_dataset)

    log_file_list = glob.glob("./output/fit/*")
    visualizer.open_tensorboard(log_file_list[-1])

    get_accuracy(model, test_dataset, class_map, draw=False)


if __name__ == "__main__":
    main()
