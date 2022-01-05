import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import uuid4
from collections import defaultdict

import numpy as np


def get_logger(name=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logging.basicConfig()
    return logger


def get_inner_ops(obj):
    ops = [v for k, v in vars(obj).items() if isinstance(v, Op)]
    return ops


@dataclass(frozen=True)
class Tensor:
    name: str
    data: Any
    trainable: bool = True

    def update(self, delta):
        data = self.data + delta
        object.__setattr__(self, 'data', data)


@dataclass(frozen=True)
class Node:
    tensor: Any
    grad_fn: Any = None

    @property
    def name(self):
        return self.tensor.name


class Graph:

    def __init__(self, entry):
        self.entry = entry
        self.nodes = []
        self._inputs = defaultdict(list)
        self._outputs = defaultdict(list)
        self._output = None

    def add_node(self, tensor, grad_fn=None):
        node = Node(tensor=tensor, grad_fn=grad_fn)
        self.nodes.append(node)

    def add_edge(self, src, dst):
        self._inputs[dst.name].append(src.name)
        self._outputs[src.name].append(dst.name)

    def compile(self):
        self.entry.assign_graph(self)

    def forward(self, *args, **kwargs):
        output = self.entry(*args, **kwargs)
        self._output = output
        return output

    def backward(self, grad=None):
        tape = {}
        if self.nodes:
            tail = self.nodes[-1]
            grad = grad or Tensor(name=tail.name, data=grad)
            tape = {tail.name: grad}
        for node in reversed(self.nodes):
            if node.grad_fn is None:
                continue
            grad = node.grad_fn(tape)
            if node.name not in tape:
                tape[node.name] = grad
            else:
                tape[node.name].update(grad.data)
        return tape


class Op(ABC):

    def __init__(self):
        self.name = f'{type(self).__name__}:{str(uuid4())[:8]}'
        self._graph = None
        self._backward_ctx = None

    def __repr__(self):
        rv = f'{type(self).__name__}(name={self.name})'
        return rv

    def __call__(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)
        if self._graph is not None:
            self._extend_graph(output, *args, **kwargs)
            self._set_backward_ctx(output, *args, **kwargs)
        return output

    def _extend_graph(self, *args, **kwargs):
        pass

    def _set_backward_ctx(self, output, *args, **kwargs):
        pass

    def assign_graph(self, graph):
        self._graph = graph
        inner_ops = get_inner_ops(self)
        for op in inner_ops:
            op.assign_graph(graph)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class Add(Op):

    def _extend_graph(self, output, a, b):
        self._graph.add_node(a, grad_fn=self._grad_a)
        self._graph.add_node(b, grad_fn=self._grad_b)
        self._graph.add_node(output)
        self._graph.add_edge(a, output)
        self._graph.add_edge(b, output)

    def _grad_a(self, tape):
        upstream = tape[self.name]
        local = 1.0
        downstream = upstream.data * local
        grad = Tensor(name=self.name, data=downstream)
        return grad

    def _grad_b(self, tape):
        upstream = tape[self.name]
        local = 1.0
        downstream = upstream.data * local
        grad = Tensor(name=self.name, data=downstream)
        return grad

    def forward(self, a, b):
        output = a.data + b.data
        output = Tensor(name=self.name, data=output)
        return output


class Mul(Op):

    def _extend_graph(self, output, a, b):
        self._graph.add_node(a, grad_fn=self._grad_a)
        self._graph.add_node(b, grad_fn=self._grad_b)
        self._graph.add_node(output)
        self._graph.add_edge(a, output)
        self._graph.add_edge(b, output)

    def _set_backward_ctx(self, output, a, b):  # noqa
        self._backward_ctx = {'a': a, 'b': b}

    def _grad_a(self, tape):
        upstream = tape[self.name]
        local = self._backward_ctx['b']
        downstream = upstream.data * local.data
        grad = Tensor(name=self.name, data=downstream)
        return grad

    def _grad_b(self, tape):
        upstream = tape[self.name]
        local = self._backward_ctx['a']
        downstream = upstream.data * local.data
        grad = Tensor(name=self.name, data=downstream)
        return grad

    def forward(self, a, b):
        output = a.data * b.data
        output = Tensor(name=self.name, data=output)
        return output


class Linear(Op):

    def __init__(self, dim_in, dim_out, weights=None, bias=None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weights = weights
        self.bias = bias

    def _init_weights_maybe(self):
        if self.weights is None:
            data = np.random.normal(size=(self.dim_in, self.dim_out))
            self.weights = Tensor(name=self.name, data=data)

    def _init_bias_maybe(self):
        if self.bias is None:
            data = np.random.normal(size=(self.dim_out,))
            self.bias = Tensor(name=self.name, data=data)

    def _set_backward_ctx(self, output, x):  # noqa
        self._backward_ctx = {'x': x}

    def _extend_graph(self, output, x):
        self._graph.add_node(x, grad_fn=self._grad_x)
        self._graph.add_node(self.weights, grad_fn=self._grad_w)
        self._graph.add_node(self.bias, grad_fn=self._grad_b)
        self._graph.add_node(output)
        self._graph.add_edge(x, output)
        self._graph.add_edge(self.weights, output)
        self._graph.add_edge(self.bias, output)

    def _grad_x(self, tape):
        upstream = tape[self.name]
        downstream = upstream.data.dot(self.weights.data.T)
        grad = Tensor(name=self.name, data=downstream)
        return grad

    def _grad_w(self, tape):
        upstream = tape[self.name]
        x = self._backward_ctx['x']
        downstream = x.data.T.dot(upstream.data)
        grad = Tensor(name=self.name, data=downstream)
        return grad

    def _grad_b(self, tape):
        upstream = tape[self.name]
        downstream = upstream.data.sum(axis=0)
        grad = Tensor(name=self.name, data=downstream)
        return grad

    def forward(self, x):
        output = x.data.dot(self.weights.data) + self.bias.data
        output = Tensor(name=self.name, data=output)
        return output


class Sigmoid(Op):

    def _set_backward_ctx(self, output, x):  # noqa
        self._backward_ctx = {'x': x, 'output': output}

    def _extend_graph(self, output, x):
        self._graph.add_node(x, grad_fn=self._grad_x)
        self._graph.add_node(output)
        self._graph.add_edge(x, output)

    def _grad_x(self, tape):
        upstream = tape[self.name]
        output = self._backward_ctx['output']
        local = output.data * (1 - output.data)
        downstream = local * upstream.data
        grad = Tensor(name=self.name, data=downstream)
        return grad

    def forward(self, x):
        output = 1 / (1 + np.exp(-x.data))
        output = Tensor(name=self.name, data=output)
        return output


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
        self._logger = logger or get_logger(type(self).__name__)

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
