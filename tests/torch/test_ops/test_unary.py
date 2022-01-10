import pytest
import torch
import numpy as np

from grad import ops
from tests.torch.backend import compare_tensors, EvalGradOp, EvalTorchOp


@pytest.mark.parametrize(
    'name, grad_op, torch_op, data',
    [
        ('relu:1d', ops.ReLU(), torch.nn.ReLU(), np.array([0.1, 0.0, 0.2, 0.3, -0.4])),
        ('relu:2d', ops.ReLU(), torch.nn.ReLU(), np.array([[0.1, 0.0, 0.2], [0.0, 0.0, -0.5]])),
        ('relu:3d', ops.ReLU(), torch.nn.ReLU(), np.array([[[0.1, 0.0, 0.2], [0.0, 0.0, -0.5]], [[2.0, 3.0, 4.0], [-2.0, -3.0, -4.0]]])),
        ('sigmoid:1d', ops.Sigmoid(), torch.nn.Sigmoid(), np.random.uniform(low=1e-5, size=(4,))),
        ('sigmoid:2d', ops.Sigmoid(), torch.nn.Sigmoid(), np.random.uniform(low=1e-5, size=(4, 2))),
        ('sigmoid:3d', ops.Sigmoid(), torch.nn.Sigmoid(), np.random.uniform(low=1e-5, size=(4, 3, 2))),
    ]
)
def test_unary_op(name, grad_op, torch_op, data):
    grad_op = EvalGradOp(grad_op)
    torch_op = EvalTorchOp(torch_op)
    grad_rv = grad_op.unary(data)
    torch_rv = torch_op.unary(data)
    output_flag = compare_tensors(grad_rv.output, torch_rv.output)
    assert output_flag
    grad_flag = compare_tensors(grad_rv.grad, torch_rv.grad)
    assert grad_flag
