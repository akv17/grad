import pytest
import torch
import numpy as np

from grad.core import Tensor, Graph
from grad.ops import (
    ReLU,
    Sigmoid
)
from tests.torch.utils import compare_tensors


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
        self._grad_eval = _GradOpEval(data=self.data, op=self.grad_op)
        self._torch_eval = _TorchOpEval(data=self.data, op=self.torch_op)
        self._grad_eval()
        self._torch_eval()

    def test_output(self):
        flag = compare_tensors(self._grad_eval.output, self._torch_eval.output)
        return flag

    def test_grad(self):
        flag = compare_tensors(self._grad_eval.grad, self._torch_eval.grad)
        return flag


@pytest.mark.parametrize(
    'name, grad_op, torch_op, data',
    [
        ('relu:1d', ReLU(), torch.nn.ReLU(), np.array([0.1, 0.0, 0.2, 0.3, -0.4])),
        ('relu:2d', ReLU(), torch.nn.ReLU(), np.array([[0.1, 0.0, 0.2], [0.0, 0.0, -0.5]])),
        ('relu:3d', ReLU(), torch.nn.ReLU(), np.array([[[0.1, 0.0, 0.2], [0.0, 0.0, -0.5]], [[2.0, 3.0, 4.0], [-2.0, -3.0, -4.0]]])),
        ('sigmoid:1d', Sigmoid(), torch.nn.Sigmoid(), np.random.uniform(low=1e-5, size=(4,))),
        ('sigmoid:2d', Sigmoid(), torch.nn.Sigmoid(), np.random.uniform(low=1e-5, size=(4, 2))),
        ('sigmoid:3d', Sigmoid(), torch.nn.Sigmoid(), np.random.uniform(low=1e-5, size=(4, 3, 2))),
    ]
)
def test_unary_op(name, grad_op, torch_op, data):
    cmd = _TestOp(data=data, grad_op=grad_op, torch_op=torch_op)
    cmd.evaluate()
    output_flag = cmd.test_output()
    assert output_flag
    grad_flag = cmd.test_grad()
    assert grad_flag
