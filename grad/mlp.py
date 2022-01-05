import numpy as np


class Linear:

    def __init__(self, dim_in, dim_out, weights=None, bias=None):
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.weights = weights
        self.bias = bias

        self.grad_weights = None
        self.grad_bias = None

        self._input = None
        self._output = None

    def forward(self, input_):
        output = input_.dot(self.weights) + self.bias
        self._input = input_
        self._output = output
        return output

    def backward(self, grad_output):
        self.grad_weights = self._input.T.dot(grad_output)
        self.grad_bias = grad_output.sum(axis=0)
        grad_input = grad_output.dot(self.weights.T)
        return grad_input


class Sigmoid:

    def __init__(self):
        self._input = None
        self._output = None

    def forward(self, input_):
        output = 1 / (1 + np.exp(-input_))
        self._input = input_
        self._output = output
        return output

    def backward(self, grad_output):
        grad_local_input = self._output * (1 - self._output)
        grad_input = grad_output * grad_local_input
        return grad_input


class MSELoss:

    def __init__(self):
        self._output = None
        self._target = None

    def compute(self, output, target):
        value = np.sum(0.5 * (target - output) ** 2)
        self._output = output
        self._target = target
        return value

    def grad(self):
        grad_input = self._output - self._target
        return grad_input


class Network:

    def __init__(self, nodes, loss):
        self.nodes = nodes
        self.loss = loss
        self._output = None

    def forward(self, input_):
        output = input_
        for node in self.nodes:
            output = node.forward(output)
        self._output = output
        return output

    def backward(self, target):
        loss_value = self.loss.compute(output=self._output, target=target)
        grad = self.loss.grad()
        for node in reversed(self.nodes):
            grad = node.backward(grad)
        return loss_value
