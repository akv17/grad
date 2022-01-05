import numpy as np

from .core import Tensor, Op

__all__ = [
    'Add',
    'Multiply',
    'Linear',
    'Sigmoid',
]


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


class Multiply(Op):

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
