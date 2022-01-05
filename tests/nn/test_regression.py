"""Following: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/"""

import unittest

import numpy as np

from grad.core import Tensor, Graph, Op
from grad.nn import Linear, Sigmoid, MSELoss, SGD, Network


class Model(Op):

    def __init__(self):
        super().__init__()
        w1 = np.array([
            [0.15, 0.25],
            [0.2,  0.3]
        ])
        w1 = Tensor(name='w1', data=w1)
        b1 = np.array([0.35, 0.35])
        b1 = Tensor(name='b1', data=b1)
        self.linear1 = Linear(dim_in=2, dim_out=2, weights=w1, bias=b1)
        self.sigmoid1 = Sigmoid()

        w2 = np.array([
            [0.4, 0.5],
            [0.45,  0.55]
        ])
        w2 = Tensor(name='w2', data=w2)
        b2 = np.array([0.6, 0.6])
        b2 = Tensor(name='b2', data=b2)
        self.linear2 = Linear(dim_in=2, dim_out=2, weights=w2, bias=b2)
        self.sigmoid2 = Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        return x


class TestTrainRegression(unittest.TestCase):

    def assert_almost_eq(self, a, b):
        diff = abs(a - b)
        self.assertLess(diff, 1e-4)

    def setUp(self) -> None:
        model = Model()
        graph = Graph(entry=model)
        graph.compile()
        loss = MSELoss()
        optimizer = SGD(learning_rate=0.5)
        self.network = Network(graph=graph, loss=loss, optimizer=optimizer)

        x = np.array([
            [0.05, 0.1],
        ])
        self.x = Tensor(name='x', data=x, trainable=False)
        y = np.array([
            [0.01, 0.99],
        ])
        self.y = Tensor(name='y', data=y, trainable=False)

    def test_train(self):
        losses = self.network.train(x=self.x, y=self.y, epochs=10)
        loss = np.mean(losses)
        self.assertLess(loss, 0.1)
