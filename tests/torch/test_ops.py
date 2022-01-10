import pytest
import torch
import numpy as np

from grad.core import Tensor, Graph
from grad.ops import (
    ReLU,
)


def _compare_tensors(t0, t1):
    t0 = t0.astype(np.float32)
    t1 = t1.astype(np.float32)
    shape_flag = t0.shape == t1.shape
    eq_flag = np.allclose(t0, t1)
    flag = shape_flag and eq_flag
    return flag


class _GradOpEval:
    _TENSOR_NAME = 'T'

    def __init__(self, data, op):
        self.data = data
        self.op = op
        self.output = None
        self.grad = None

    def __call__(self):
        tensor = Tensor(name=self._TENSOR_NAME, data=self.data)
        graph = Graph(self.op)
        graph.compile()
        output = graph.forward(tensor)
        grads = graph.backward()
        grad = grads[self._TENSOR_NAME]
        grad = grad
        self.output = output.data
        self.grad = grad.data


class _TorchOpEval:

    def __init__(self, data, op):
        self.data = data
        self.op = op
        self.output = None
        self.grad = None

    def __call__(self):
        shape = self.data.shape
        tensor = torch.as_tensor(self.data)
        tensor.requires_grad = True
        grad = torch.as_tensor(np.ones(shape))
        output = self.op(tensor)
        output.backward(grad)
        grad = tensor.grad
        self.output = output.detach().numpy()
        self.grad = grad.detach().numpy()


class _TestOp:

    def __init__(self, data, grad_op, torch_op):
        self.data = data
        self.grad_op = grad_op
        self.torch_op = torch_op
        self._grad_eval = None
        self._torch_eval = None

    def evaluate(self):
        self._grad_eval = _GradOpEval(data=self.data, op=ReLU())
        self._torch_eval = _TorchOpEval(data=self.data, op=torch.nn.ReLU())
        self._grad_eval()
        self._torch_eval()

    def test_output(self):
        flag = _compare_tensors(self._grad_eval.output, self._torch_eval.output)
        return flag

    def test_grad(self):
        flag = _compare_tensors(self._grad_eval.grad, self._torch_eval.grad)
        return flag


def _test_op(data, grad_op, torch_op):
    cmd = _TestOp(data=data, grad_op=grad_op, torch_op=torch_op)
    cmd.evaluate()
    output_flag = cmd.test_output()
    assert output_flag
    grad_flag = cmd.test_grad()
    assert grad_flag


@pytest.mark.parametrize(
    'data',
    [
        np.array([0.1, 0.0, 0.2, 0.3, -0.4]),
        np.array([[0.1, 0.0, 0.2], [0.0, 0.0, -0.5]]),
        np.array([[[0.1, 0.0, 0.2], [0.0, 0.0, -0.5]], [[2.0, 3.0, 4.0], [-2.0, -3.0, -4.0]]]),
    ]
)
def test_relu(data):
    _test_op(data=data, grad_op=ReLU(), torch_op=torch.nn.ReLU())
