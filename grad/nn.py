import logging

import numpy as np

from .core import Tensor
from .ops import (
    Add,
    Multiply,
    Linear,
    Sigmoid
)

__all__ = [
    'Add',
    'Multiply',
    'Linear',
    'Sigmoid',
    'MSELoss',
    'SGD',
    'Network'
]


def _get_logger(name=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logging.basicConfig()
    return logger


class MSELoss:

    def __init__(self):
        self._output = None
        self._target = None
        self.name = type(self).__name__

    def compute(self, output, target):
        value = np.sum(0.5 * (target.data - output.data) ** 2)
        value = Tensor(name=self.name, data=value)
        self._output = output
        self._target = target
        return value

    def grad(self):
        data = self._output.data - self._target.data
        grad = Tensor(name=self.name, data=data)
        return grad


class SGD:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self._graph = None

    def assign_graph(self, graph):
        self._graph = graph

    def step(self, grads):
        updated = set()
        for node in self._graph.nodes:
            grad = grads.get(node.name)
            if not node.tensor.trainable or node.name in updated or grad is None:
                continue
            tensor = node.tensor
            delta = -(grad.data * self.learning_rate)
            tensor.update(delta)
            updated.add(node.name)


class Network:

    def __init__(self, graph, loss, optimizer, logger=None):
        self.graph = graph
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer.assign_graph(self.graph)
        self._output = None
        self._logger = logger or _get_logger(type(self).__name__)

    def forward(self, x):
        output = self.graph.forward(x)
        self._output = output
        return output

    def backward(self, target):
        loss_value = self.loss.compute(output=self._output, target=target)
        grad = self.loss.grad()
        grads = self.graph.backward(grad)
        data = {'loss': loss_value, 'grads': grads}
        return data

    def step(self, grads):
        self.optimizer.step(grads)

    def train(self, x, y, epochs=1, batch_size=64, verbose=None):
        iter_num = 0
        losses = []
        for epoch in range(epochs):
            for i in range(0, len(x.data), batch_size):
                x_batch = x.data[i: i + batch_size]
                x_batch = Tensor(name='x', data=x_batch, trainable=False)
                y_batch = y.data[i: i + batch_size]
                y_batch = Tensor(name='y', data=y_batch, trainable=False)
                self.forward(x_batch)
                backward_data = self.backward(y_batch)
                loss = backward_data['loss'].data
                losses.append(loss)
                grads = backward_data['grads']
                self.step(grads)
                if verbose is not None and (iter_num + 1) % verbose == 0:
                    msg = (
                        f'epoch: {epoch+1}\t'
                        f'iter: {iter_num+1}\t'
                        f'loss: {np.mean(losses)}\t'
                        f'batch_loss: {loss}'
                    )
                    self._logger.info(msg)
                iter_num += 1
                self.graph.reset()
        return losses
