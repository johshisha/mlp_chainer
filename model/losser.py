#coding: utf-8

import chainer
from chainer import computational_graph, serializers, cuda, optimizers, Variable
from chainer import functions as F, links as L
import numpy as np
from IPython import embed

class Losser(chainer.Chain):
    def __init__(self,model):
        super(Losser,self).__init__(model=model)
    def __call__(self,x,t):
        y = self.model(x)
        self.loss = F.softmax_cross_entropy(y, t)
        self.acc = F.accuracy(y, t)
        return self.loss, self.acc

