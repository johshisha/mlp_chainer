

from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from chainer import computational_graph as c
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




# draw a image of handwriting number
def draw_digit_ae(data, n, row, col, _type, size = 28):
    plt.subplot(row, col, n)
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]                 # flip vertical
    plt.xlim(0,size)
    plt.ylim(0,size)
    plt.pcolor(Z)
    plt.title("type=%s"%(_type), size=8)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

def draw_digits(y, t):
    plt.figure(figsize=(15,25))

    num = y.data.shape[0]
    for i in range(int(num/10)):
        for j in range (10):
            img_no = i*10+j
            pos = (2*i)*10+j
            draw_digit_ae(t[img_no].data,  pos+1, int(num/10)*2, 10, "ans")

        for j in range (10):
            img_no = i*10+j
            pos = (2*i+1)*10+j
            draw_digit_ae(y[img_no].data, pos+1, int(num/10)*2, 10, "pred")

    plt.show()


def draw_graph(variable):
    g = c.build_computational_graph(variable)
    with open('resource/g.out', 'w') as o:
        o.write(g.dump())