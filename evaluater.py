#coding: utf-8

import chainer
from chainer import computational_graph, serializers, cuda, optimizers, Variable
from chainer import functions as F, links as L
import numpy as np
import random, argparse
from IPython import embed


from util import util, drawer
from model import MLP


archs = {
    'normal': MLP.MLP
}


def calc_acc(y, t):
    from sklearn.metrics import classification_report
    print(classification_report(y, t))



def show_acc(model, x_test, y_test, batchsize):
    corrects = []
    estimates = []
    #show test sample estimation
    for data, label in zip(x_test, y_test):
        x, t = Variable(data.reshape(1,data.shape[0])), Variable(np.array([label]))
        y = model(x if batchsize != 1 else F.reshape(x, (batchsize, x.data.shape[0])))

        #embed()
        corrects.extend(cuda.to_cpu(t.data))
        estimates.extend(np.array([cuda.to_cpu(y.data).argmax()]))

    calc_acc(estimates, corrects)


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

    show_acc(model, x_test, y_test, batchsize)


