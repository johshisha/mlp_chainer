#coding: utf-8

import chainer
from chainer import computational_graph as c, serializers, cuda, optimizers, Variable
from chainer import functions as F, links as L
import numpy as np
import argparse
from IPython import embed
import random


from util import util
from model import MLP
from model.losser import Losser


archs = {
    'normal': MLP.MLP
}


n_epoch = 10
batchsize = 32

def train(model, optimizer, x_train, y_train, x_test, y_test, shuffle=True):
    n_train = len(x_train)
    n_test = len(x_test)
    losser = Losser(model)
    epoch_loss = []

    if shuffle:
        x_train, y_train = shuffle_data(x_train, y_train)

    for epoch in range(n_epoch):
        sum_loss = np.float32(0)
        sum_acc = np.float32(0)
        for i in range(0, n_train, batchsize):
            x_batch = Variable(x_train[i:i+batchsize])
            y_batch = Variable(y_train[i:i+batchsize])


            loss, acc = losser(x_batch, y_batch)
            # print(loss.data)
            optimizer.zero_grads()  #backwardの直前におく！！！！！！！！！！！！！！
            loss.backward()
            optimizer.update()

            sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
            sum_acc += float(cuda.to_cpu(acc.data)) * batchsize

        sum_loss /= (i+batchsize)
        sum_acc /= (i+batchsize)
        print('epoch %d done, epoch loss is %f, acc is %f'%(epoch, sum_loss, sum_acc))
        epoch_loss.append(sum_loss)


        if epoch%2 == 0:
            test_loss = np.float32(0)
            test_acc = np.float32(0)
            for i in range(0, n_test, batchsize):
                x_batch = Variable(x_test[i:i+batchsize])
                y_batch = Variable(y_test[i:i+batchsize])


                loss, acc = losser(x_batch, y_batch)

                test_loss += float(cuda.to_cpu(loss.data)) * batchsize
                test_acc += float(cuda.to_cpu(acc.data)) * batchsize

            test_loss /= (i+batchsize)
            test_acc /= (i+batchsize)
            print('test:: epoch loss is %f, acc is %f'%(test_loss, test_acc))


    return model

def shuffle_data(x, y):
    d = list(zip(x,y))
    random.shuffle(d)
    x = np.array([x[0] for x in d])
    y = np.array([x[1] for x in d])
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='normal',
                        help='Auto-encoder architecture')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--beta1', '-be', type=float, default=0.5,
                        help='Beta1 in Adam parameter')
    parser.add_argument('--out', '-o', default='resource',
                        help='Output directory')
    parser.add_argument('--no_dropout', action='store_true')
    parser.set_defaults(no_dropout=False)
    args = parser.parse_args()


    print('learning %s multi layer perceptron'%args.arch)

    save_model = '%s/%s.model'%(args.out, args.arch)
    n_epoch = args.epoch
    batchsize = args.batchsize

    #difine model and optimizer
    model = archs[args.arch]()
    if args.gpu >= 0:
        xp = cuda.cupy
        cuda.get_device(args.gpu)
        model.to_gpu()
    else:
        xp = np
        model.to_cpu()

    x_train, x_test, y_train, y_test = list(map(xp.array, util.load_mnist(noised=False))) #load mnist data


    if args.no_dropout:
        model.train = False  #without dropout
    optimizer = optimizers.Adam(alpha=0.01, beta1=args.beta1)
    optimizer.setup(model)

    model = train(model, optimizer, x_train, y_train, x_test, y_test)

    serializers.save_npz(save_model, model)


