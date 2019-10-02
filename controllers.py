"""
controllers.py
ResponderAPIを生成し、コントローラを管理するファイル。
"""

import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.datasets import mnist
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import scale
import uuid

import matplotlib.pyplot as plt
import numpy as np
import datetime

import os

import responder

api = responder.API(
    title='Let\'s Learn Machine Learning',
    openapi='3.0.2',
    description='Web上で機械学習を学ぼう！',
    version='Beta',
)

# 学習結果
result = list()


class IndexController:
    async def on_get(self, req, resp):
        title = 'ResponderとKerasで学ぶ機械学習アプリケーション'
        resp.html = api.template('index.html', title=title)


class CreateController:
    async def on_get(self, req, resp, dataset):

        if dataset not in ['mnist', 'iris', 'wine']:
            api.redirect(resp, '/404')
            return

        title = 'ネットワークを作成【' + dataset + '】'
        # 入力サイズ
        input_length = 784 if dataset == 'mnist' else 4 if dataset == 'iris' else 13
        # 出力サイズ(クラス数)
        output_length = 10 if dataset == 'mnist' else 3

        resp.html = api.template('create.html',
                                 dataset=dataset,
                                 input_length=input_length,
                                 output_length=output_length,
                                 title=title)


class LearnController:
    async def on_get(self, req, resp, dataset):
        api.redirect(resp, '/404')
        return

    async def on_post(self, req, resp, dataset):
        # 任意のIDを設定する
        uid = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')
        uid += str(uuid.uuid4())

        data = await req.media()

        # [あとで実装] 学習データセットを取得
        train_data = get_data(dataset)

        # [あとで実装] バックグラウンドで学習させる
        learn_model(data, train_data, uid)

        # 学習終了待機ページへリダイレクト
        api.redirect(resp, '/learning/' + uid)


class LearningController:
    async def on_get(self, req, resp, uid):
        img_path = 'static/images/' + uid + '_history.svg'

        if not os.path.isfile(img_path):
            title = 'ネットワークを学習中...'
            resp.html = api.template('learning.html', title=title, uid=uid)

        else:
            svg = open(img_path, 'r').readlines()[4:]

            # svgファイルを削除する場合
            # os.remove(img_path)

            title = 'ネットワークの学習が完了しました'
            resp.html = api.template('result.html',
                                     title=title,
                                     uid=uid,
                                     svg=svg,
                                     result=result)


class NotFoundController:
    async def on_get(self, req, resp):
        title = '404 Not Found'
        resp.html = api.template('404.html', title=title)


@api.background.task
def learn_model(data, all_data, uid):

    fc = None
    if 'fc[]' in data:  # 中間層がPOSTデータにあるならば
        fc = data.get_list('fc[]')

    # モデル構築
    model = Sequential()

    if fc is None:  # 中間層がないならば、シンプルな2層ネットワーク
        model.add(Dense(int(data['output']), activation='softmax', input_shape=(int(data['input']),)))

    else:  # 中間層がある
        is_first = True

        for _fc in fc:
            if is_first:
                model.add(Dense(int(_fc), activation='sigmoid', input_shape=(int(data['input']),)))
                is_first = False
            else:
                model.add(Dense(int(_fc), activation='sigmoid'))

        model.add(Dense(int(data['output']), activation='softmax'))

    # モデル作成おわり
    # コンソールにネットワーク情報を表示させたい場合
    # model.summary()

    # 学習
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(  # 学習率の最適化部分
                      lr=float(data['eta']),  # 初期学習率
                      decay=float(data['decay']),  # 学習率減衰
                      momentum=float(data['momentum']),  # 慣性項
                  ),
                  metrics=['accuracy'])

    # 学習結果
    history = model.fit(all_data['x_train'], all_data['y_train'],  # 画像とラベルデータ
                        epochs=int(data['epoch']),  # エポック数の指定
                        verbose=1,  # ログ出力の指定. 0だとログが出ない
                        validation_data=(all_data['x_test'], all_data['y_test']))

    acc = history.history['acc']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']

    # モデルを破棄
    keras.backend.clear_session()

    # 結果をグローバル変数として保持
    global result
    result = [acc, loss, val_acc, val_loss]

    # ここからグラフ描画部分
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='acc', color='b')
    plt.plot(val_acc, label='val_acc', color='g')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='loss', color='b', ls='--')
    plt.plot(val_loss, label='val_loss', color='g', ls='--')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.savefig('static/images/' + uid + '_history.svg', dpi=300)
    plt.close()


def get_data(dataset):
    train_data = dict()

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)  # 2次元配列を1次元に変換
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')  # int型をfloat32型に変換
        x_test = x_test.astype('float32')
        x_train /= 255  # [0-255]の値を[0.0-1.0]に変換
        x_test /= 255

        train_data['x_train'] = x_train
        train_data['x_test'] = x_test

        # One-hot ベクタに変換
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        train_data['y_train'] = y_train
        train_data['y_test'] = y_test

    elif dataset == 'iris':
        iris = load_iris()
        all_data = scale(iris['data'])  # 平均0 分散1 で標準化
        target = iris['target']

        # データを対応関係を保ったままシャッフル
        data_size = len(all_data)
        p = np.random.permutation(data_size)
        all_data = all_data[p]
        target = target[p]
        target = keras.utils.np_utils.to_categorical(target)  # to one-hot

        # 訓練データと検証データに分割する
        train_data['x_train'] = all_data[:120]
        train_data['x_test'] = all_data[120:]
        train_data['y_train'] = target[:120]
        train_data['y_test'] = target[120:]

    else:
        wine = load_wine()
        all_data = scale(wine['data'])  # 平均0 分散1 で標準化
        target = wine['target']

        # データを対応関係を保ったままシャッフル
        data_size = len(all_data)
        p = np.random.permutation(data_size)
        all_data = all_data[p]
        target = target[p]
        target = keras.utils.np_utils.to_categorical(target)  # to one-hot

        # 訓練データと検証データに分割する
        train_data['x_train'] = all_data[:120]
        train_data['x_test'] = all_data[120:]
        train_data['y_train'] = target[:120]
        train_data['y_test'] = target[120:]

    return train_data
