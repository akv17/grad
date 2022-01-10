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
class EvalResultUnary:
    output: Any
    grad: Any


@dataclass(frozen=True)
class EvalResultBinary:
    output: Any
    grad_a: Any
    grad_b: Any


class EvalGradOp:

    def __init__(self, op):
        self.op = op

    def unary(self, x):
        tensor = Tensor(name='x', data=x)
        graph = Graph(self.op)
        graph.compile()
        output = graph.forward(tensor)
        grads = graph.backward()
        grad = grads['x']
        rv = EvalResultUnary(output=output.data, grad=grad.data)
        return rv

    def binary(self, a, b):
        a = Tensor(name='a', data=a)
        b = Tensor(name='b', data=b)
        grad = Tensor(name='b', data=np.ones(a.data.shape))
        graph = Graph(self.op)
        graph.compile()
        output = graph.forward(a, b)
        grads = graph.backward(grad=grad)
        grad_a = grads['a']
        grad_b = grads['b']
        rv = EvalResultBinary(
            output=output.data,
            grad_a=grad_a.data,
            grad_b=grad_b.data
        )
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
        rv = EvalResultUnary(output=output.detach().numpy(), grad=grad.detach().numpy())
        return rv

    def binary(self, a, b):
        a = torch.as_tensor(a)
        b = torch.as_tensor(b)
        grad = torch.as_tensor(np.ones(a.shape))
        a.requires_grad = True
        b.requires_grad = True
        output = self.op(a, b)
        output.backward(grad)
        rv = EvalResultBinary(
            output=output.detach().numpy(),
            grad_a=a.grad.detach().numpy(),
            grad_b=b.grad.detach().numpy(),
        )
        return rv
