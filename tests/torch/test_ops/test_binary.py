import pytest
import torch
import numpy as np

from grad import ops
from tests.torch.backend import compare_tensors, EvalGradOp, EvalTorchOp


@pytest.mark.parametrize(
    'name, grad_op, torch_op, a, b',
    [
        ('add:1d', ops.Add(), torch.add, np.random.normal(size=(4,)), np.random.normal(size=(4,))),
        ('add:2d', ops.Add(), torch.add, np.random.normal(size=(4, 2)), np.random.normal(size=(4, 2))),
        ('add:3d', ops.Add(), torch.add, np.random.normal(size=(4, 3, 2)), np.random.normal(size=(4, 3, 2))),
        ('mul:1d', ops.Multiply(), torch.mul, np.random.normal(size=(4,)), np.random.normal(size=(4,))),
        ('mul:2d', ops.Multiply(), torch.mul, np.random.normal(size=(4, 2)), np.random.normal(size=(4, 2))),
        ('mul:3d', ops.Multiply(), torch.mul, np.random.normal(size=(4, 3, 2)), np.random.normal(size=(4, 3, 2))),
    ]
)
def test_binary_op(name, grad_op, torch_op, a, b):
    grad_op = EvalGradOp(grad_op)
    torch_op = EvalTorchOp(torch_op)
    grad_rv = grad_op.binary(a, b)
    torch_rv = torch_op.binary(a, b)
    output_flag = compare_tensors(grad_rv.output, torch_rv.output)
    assert output_flag
    grad_a_flag = compare_tensors(grad_rv.grad_a, torch_rv.grad_a)
    assert grad_a_flag
    grad_b_flag = compare_tensors(grad_rv.grad_b, torch_rv.grad_b)
    assert grad_b_flag
