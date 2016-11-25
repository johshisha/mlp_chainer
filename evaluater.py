#coding: utf-8

import chainer
from chainer import computational_graph, serializers, cuda, optimizers, Variable
from chainer import functions as F, links as L
import numpy as np
import random, argparse


from util import util
from model import MLP, deep_auto_encoder, cnn_auto_encoder


archs = {
    'normal': MLP.AutoEncoder,
    'deep': deep_auto_encoder.DeepAutoEncoder,
    'cnn': cnn_auto_encoder.CnnAutoEncoder
}



def show_reconstruction(model, x_test, y_test, batchsize):
    #show test sample estimation
    indexs = random.sample(range(len(y_test)), batchsize)#抽出する添字を取得
    x, t = Variable(x_test[indexs]), Variable(y_test[indexs])
    y = model(x if batchsize != 1 else F.reshape(x, (batchsize, x.data.shape[0])))
    util.draw_digits(y, t)


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
    parser.add_argument('--out', '-o', default='resource',
                        help='Output directory')
    args = parser.parse_args()


    saved_model = '%s/%s.model'%(args.out, args.arch)
    batchsize = args.batchsize

    x_train, x_test, y_train, y_test = util.load_mnist(noised=False) #load mnist data


    #difine model and optimizer
    model = archs[args.arch]()
    if args.gpu >= 0:
        model.to_gpu()
    else:
        model.to_cpu()

    serializers.load_npz(saved_model, model)
    model.train = False

    show_reconstruction(model, x_test, y_test, batchsize)


