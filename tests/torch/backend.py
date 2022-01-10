from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from grad.core import Tensor, Graph


def compare_tensors(t0, t1):
    t0 = t0.astype(np.float32)
    t1 = t1.astype(np.float32)
    shape_flag = t0.shape == t1.shape
    eq_flag = np.allclose(t0, t1)
    flag = shape_flag and eq_flag
    return flag


@dataclass(frozen=True)
class EvalResult:
    output: Any
    grad: Any


class EvalGradOp:
    _TENSOR_NAME = 'T'

    def __init__(self, op):
        self.op = op

    def unary(self, x):
        tensor = Tensor(name=self._TENSOR_NAME, data=x)
        graph = Graph(self.op)
        graph.compile()
        output = graph.forward(tensor)
        grads = graph.backward()
        grad = grads[self._TENSOR_NAME]
        grad = grad
        rv = EvalResult(output=output.data, grad=grad.data)
        return rv


class EvalTorchOp:

    def __init__(self, op):
        self.op = op

    def unary(self, x):
        tensor = torch.as_tensor(x)
        tensor.requires_grad = True
        grad = torch.as_tensor(np.ones(x.shape))
        output = self.op(tensor)
        output.backward(grad)
        grad = tensor.grad
        rv = EvalResult(output=output.detach().numpy(), grad=grad.detach().numpy())
        return rv
