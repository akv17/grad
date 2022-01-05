import unittest
from grad.core import Tensor, Graph, Op
from grad.ops import Add, Multiply


class MyOp(Op):

    def __init__(self):
        super().__init__()
        self.add = Add()
        self.mul = Multiply()

    def forward(self, x, y, z):
        q = self.add(x, y)
        p = self.mul(q, z)
        return p


class TestSimpleGraph(unittest.TestCase):

    def setUp(self) -> None:
        op = MyOp()
        graph = Graph(entry=op)
        graph.compile()
        self.graph = graph
        self.x = Tensor(name='Input:x', data=-2.0)
        self.y = Tensor(name='Input:y', data=5.0)
        self.z = Tensor(name='Input:z', data=-4.0)

    def test_forward(self):
        output = self.graph.forward(x=self.x, y=self.y, z=self.z)
        assert isinstance(output, Tensor)
        assert output.data == -12.0, output.data

    def test_backward(self):
        self.graph.forward(x=self.x, y=self.y, z=self.z)
        grads = self.graph.backward()
        assert grads['Input:x'].data == -4.0
        assert grads['Input:y'].data == -4.0
        assert grads['Input:z'].data == 3.0
