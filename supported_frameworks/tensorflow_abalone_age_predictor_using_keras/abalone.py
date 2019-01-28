# -*- coding: utf-8 -*-

import numpy as np
import os
import tensorflow as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput

INPUT_TENSOR_NAME = "inputs"
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001


def model_fn(features, labels, mode, params):
    """Estimator のためのモデル定義メソッド
     # メソッドの構成は以下のとおり
     # 1. Keras の Functional API 経由でモデルの設定を記述
     # 2. Tensorflow を使って，学習・評価時の損失関数を定義
     # 3. Tensorflow を使って，学習時のオペレータ・オプティマイザを定義
     # 4. Tensorflow の tensors として予測値を取得
     # 5. 評価用のメトリクスを生成
     # 6. 予測値・損失関数・学習オペレータ・評価用メトリクスを EstimatorSpec オブジェクトとして返す"""

    # 1. Keras の Functional API 経由でモデルの設定を記述

    first_hidden_layer = tf.keras.layers.Dense(10, activation='relu', name='first-layer')(features[INPUT_TENSOR_NAME])
    second_hidden_layer = tf.keras.layers.Dense(10, activation='relu')(first_hidden_layer)
    output_layer = tf.keras.layers.Dense(1, activation='linear')(second_hidden_layer)

    predictions = tf.reshape(output_layer, [-1])

    # 予測モードのとき（= `ModeKeys.PREDICT`）は，こちらの EstimatorSpec
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"ages": predictions},
            export_outputs={SIGNATURE_NAME: PredictOutput({"ages": predictions})})

    # 2. Tensorflow を使って，学習・評価時の損失関数を定義
    loss = tf.losses.mean_squared_error(labels, predictions)

    # 3. Tensorflow を使って，学習時のオペレータ・オプティマイザを定義
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer="SGD")

    # 4. Tensorflow の tensors として予測値を取得
    predictions_dict = {"ages": predictions}

    # 5. 評価用のメトリクスを生成
    # RMSE を追加のメトリックとして計算
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float32), predictions)
    }

    # 予測値・損失関数・学習オペレータ・評価用メトリクスを EstimatorSpec オブジェクトとして返す
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def serving_input_fn(params):
    tensor = tf.placeholder(tf.float32, shape=[1, 7])
    return build_raw_serving_input_receiver_fn({INPUT_TENSOR_NAME: tensor})()


params = {"learning_rate": LEARNING_RATE}


def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'abalone_train.csv')


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'abalone_test.csv')


def _input_fn(training_dir, training_filename):
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, training_filename), target_dtype=np.int, features_dtype=np.float32)

    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)()
