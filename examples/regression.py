"""
Here we build and train simple neural net to solve 1-dim regression task.
Network architecture:
    1. Input: N x 10
    2. LinearLayer1: N x 16
    3. Sigmoid
    4. LinearLayer2: N x 8
    5. Sigmoid
    6. LinearLayer3: N x 4
    7. Sigmoid
    8. Output: N x 1
"""

import numpy as np

from grad.core import Tensor, Graph, Op
from grad.nn import Linear, Sigmoid, MSELoss, SGD, Network


class Model(Op):
    """Defines neural net."""

    def __init__(self):
        super().__init__()
        self.linear1 = Linear(dim_in=10, dim_out=16, name='linear1')
        self.sigmoid1 = Sigmoid(name='sigmoid1')

        self.linear2 = Linear(dim_in=16, dim_out=8, name='linear2')
        self.sigmoid2 = Sigmoid(name='sigmoid2')

        self.linear3 = Linear(dim_in=8, dim_out=4, name='linear3')
        self.sigmoid3 = Sigmoid(name='sigmoid3')

        self.linear4 = Linear(dim_in=4, dim_out=1, name='linear4')

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        x = self.linear3(x)
        x = self.sigmoid3(x)
        x = self.linear4(x)
        return x


def _make_regression(samples=100, features=10):
    coef = np.random.normal(size=(features, 1))
    x = np.random.normal(size=(samples, features))
    y = x.dot(coef)
    y /= y.max()
    return x, y


def main():
    # instantiate model and its computational graph.
    model = Model()
    graph = Graph(entry=model)
    graph.compile()
    # instantiate MSE loss.
    loss = MSELoss()
    # instantiate SGD optimizer.
    optimizer = SGD(learning_rate=0.1)
    # instantiate neural network object.
    network = Network(graph=graph, loss=loss, optimizer=optimizer)

    # generate synthetic dataset.
    x, y = _make_regression()
    x = Tensor(name='x', data=x, trainable=False)
    y = Tensor(name='y', data=y, trainable=False)

    # train the net logging loss every 10 steps.
    network.train(x, y, batch_size=8, epochs=100, verbose=10)


if __name__ == '__main__':
    main()
