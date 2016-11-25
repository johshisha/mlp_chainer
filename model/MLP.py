#coding: utf-8

import chainer
from chainer import computational_graph, serializers, cuda, optimizers, Variable
from chainer import functions as F, links as L
import numpy as np


n_input = 784
n_units = 1000
n_out = 10

class MLP(chainer.Chain):
    train = True
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h = F.dropout(F.relu(self.l1(x)), train=self.train)
        h = F.dropout(F.relu(self.l2(h)), train=self.train)
        y = self.l3(h)
        return y

