

from sklearn.datasets import fetch_mldata
import numpy as np
import random

def load_mnist(N = 60000, noised=True):
    # MNISTの手書き数字データのダウンロード
    # #HOME/scikit_learn_data/mldata/mnist-original.mat にキャッシュされる
    print('fetch MNIST dataset')
    mnist = fetch_mldata('MNIST original')

    # mnist.data : 70,000件の784次元ベクトルデータ
    mnist.data   = mnist.data.astype(np.float32)
    mnist.data  /= 255     # 0-1のデータに変換

    np.random.seed(0)
    np.random.shuffle(mnist.data)

    # mnist.target : 正解データ（教師データ）
    mnist.target = mnist.target.astype(np.int32)
    np.random.seed(0)
    np.random.shuffle(mnist.target)
    y_train, y_test = np.split(mnist.target, [N])


    # 学習用データを N個、検証用データを残りの個数と設定
    if noised:
        # Add noise
        noise_ratio = 0.2
        for data in mnist.data:
            perm = np.random.permutation(mnist.data.shape[1])[:int(mnist.data.shape[1]*noise_ratio)]
            data[perm] = 0.0

    x_train, x_test = np.split(mnist.data,   [N])

    return x_train, x_test, y_train, y_test
