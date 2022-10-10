import datetime
import os
import glob
import os.path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from modules.dataset import ModelNetDataset
from modules import visualizer
import open3d as o3d


def conv_bn(x, filters):
    x = keras.layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    return keras.layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = keras.layers.Dense(filters)(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    return keras.layers.Activation("relu")(x)


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


def tnet(inputs, num_features):
    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = keras.layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=lambda mat: orthogonal_regularizer(mat, num_features),
    )(x)
    feat_T = keras.layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return keras.layers.Dot(axes=(2, 1))([inputs, feat_T])


def point_net(point_num, class_num):
    """
    Functional API で PointNet を定義.
    (cf. https://dev.classmethod.jp/articles/tensorflow-keras-api-pattern/)

    :param point_num:
    :param class_num:
    :return:
    """
    encoder_inputs = keras.Input(shape=(point_num, 3))
    # 3次元空間の正準座標を求めて適用
    x = tnet(encoder_inputs, 3)
    # 特徴量抽出
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    # 特徴量空間の正準座標を求めて適用
    x = tnet(x, 32)
    # 近似的な対称関数を適用 (混合してから対称化)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    encoder_outputs = keras.layers.GlobalMaxPooling1D()(x)
    encoder_model = keras.Model(encoder_inputs, encoder_outputs, name="pointnet_encoder")

    # クラス分類
    classifier_inputs = keras.Input(shape=(512,))
    x = dense_bn(classifier_inputs, 256)
    x = keras.layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = keras.layers.Dropout(0.3)(x)
    classifier_outputs = keras.layers.Dense(class_num, activation="softmax")(x)
    classifier_model = keras.Model(classifier_inputs, classifier_outputs, name="pointnet_classifier")

    inputs = keras.Input(shape=(point_num, 3))
    model = keras.Model(inputs, classifier_model(encoder_model(inputs)), name="point_net")
    model.summary()

    return model


def fit_tf_model(model, train_dataset, test_dataset):
    input_num = model.input_shape[-1]
    class_num = model.output_shape[-1]
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )

    log_dir = f"./output/fit/PointNet_ModelNet{class_num}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(train_dataset, epochs=20, validation_data=test_dataset,
              callbacks=[tensorboard_callback])

    pretrained_name = f"./models/PointNet_{input_num}_{class_num}"
    model.save(pretrained_name)



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
    max_num, max_diff = 1024, 128
    class_num = 40
    dataset = ModelNetDataset(class_num, cache=True)
    train_dataset, test_dataset, class_map = dataset.get_tf_dataset(max_num, max_diff, mask=False)

    pretrained_name = f"./models/PointNet_{max_num}_{class_num}"
    if not os.path.exists(pretrained_name):
        model = point_net(max_num, class_num)
        fit_tf_model(model, train_dataset, test_dataset)
    else:
        model = keras.models.load_model(pretrained_name)

    log_file_list = glob.glob("./output/fit/*")
    visualizer.open_tensorboard(log_file_list[-1])

    get_accuracy(model, test_dataset, class_map, draw=False)


if __name__ == "__main__":
    main()
