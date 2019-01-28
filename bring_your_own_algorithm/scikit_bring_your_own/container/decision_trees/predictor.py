# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback

import flask

import pandas as pd

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# モデルを提供するためのシングルトンクラス
# get_model() メソッドでモデルデータをロードし，predict() メソッドでロードしたモデルの推論を実行します

class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        if cls.model == None:
            with open(os.path.join(model_path, 'decision-tree-model.pkl'), 'r') as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        clf = cls.get_model()
        return clf.predict(input)

    
# 推論処理を行うための flask アプリケーション
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    # コンテナが正常に動作しているかを判定するメソッド．ここでは，モデルをロードできたら正常と返すようにしています
    health = ScoringService.get_model() is not None

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    # csv を受け取って pandas のデータフレームに変換し，モデルで推論．結果を csv に戻してから返します
    data = None

    # csv を pandas に変換
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO.StringIO(data)
        data = pd.read_csv(s, header=None)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    # 推論を実施
    predictions = ScoringService.predict(data)

    # 結果を csv に戻してから返却
    out = StringIO.StringIO()
    pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()
    return flask.Response(response=result, status=200, mimetype='text/csv')
