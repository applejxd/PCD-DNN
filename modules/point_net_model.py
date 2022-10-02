import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
import os


def conv_bn(inputs, filters):
    """
    バッチ正則化付き pointwise 畳込み層
    """
    x, mask = tf.split(inputs, num_or_size_splits=[-1, 1], axis=-1)
    x = keras.layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    x = keras.layers.Activation("relu")(x)
    return tf.concat([x, mask], axis=-1)


def dense_bn(inputs, filters):
    """
    バッチ正則化付き全結合層
    """
    x, mask = tf.split(inputs, num_or_size_splits=[-1, 1], axis=-1)
    x = keras.layers.Dense(filters)(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    x = keras.layers.Activation("relu")(x)
    return tf.concat([x, mask], axis=-1)


def masked_max_pooling(inputs):
    """
    mask 適用マックスプーリング層
    """
    x, mask = tf.split(inputs, num_or_size_splits=[-1, 1], axis=-1)
    filters = tf.unstack(tf.shape(x))[-1]
    # Extend mask to match net output dimension
    mask = tf.tile(mask, multiples=[1, 1, filters])
    x = tf.where(tf.equal(mask, 1.0), x, tf.fill(tf.shape(x), -np.inf))
    return keras.layers.GlobalMaxPooling1D()(x)


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
    """
    T-net
    :param inputs: 入力データ
    :param num_features: 回転する次元数
    :return:
    """
    # 近似的な対称関数を適用 (変換してから対称化)
    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = masked_max_pooling(x)

    # 直交行列を学習
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = keras.layers.Dense(
        num_features * num_features,  # 直交行列の成分数
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=lambda mat: orthogonal_regularizer(mat, num_features),
    )(x)
    feature_mat = keras.layers.Reshape((num_features, num_features))(x)

    # Apply affine transformation to input features
    x, mask = tf.split(inputs, num_or_size_splits=[-1, 1], axis=-1)
    x = keras.layers.Dot(axes=(2, 1))([x, feature_mat])
    return tf.concat([x, mask], axis=-1)


def point_net(input_num, output_num):
    """
    Functional API で PointNet を定義.
    (cf. https://dev.classmethod.jp/articles/tensorflow-keras-api-pattern/)

    :param input_num:
    :param output_num:
    :return:
    """
    inputs = keras.Input(shape=(input_num, 4))

    # 3次元空間の正準座標を求めて適用
    x = tnet(inputs, 3)
    # 特徴量抽出
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    # 特徴量空間の正準座標を求めて適用
    x = tnet(x, 32)

    # 近似的な対称関数を適用 (混合してから対称化)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = masked_max_pooling(x)

    # クラス分類
    x = dense_bn(x, 256)
    x = keras.layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = keras.layers.Dropout(0.3)(x)

    # 確率出力
    outputs = keras.layers.Dense(output_num, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="point_net")
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
