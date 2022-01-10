"""
Here we define, evaluate and differentiate scalar function `F(x, y, z) = (x + y) * z`.
Following: https://youtu.be/dB-u77Y5a6A?t=534.
"""

from grad.core import Tensor, Graph, Op
from grad.ops import Add, Multiply


class F(Op):
    """Defines function `F(x, y, z) = (x + y) * z`."""

    def __init__(self):
        super().__init__()
        self.add = Add()
        self.mul = Multiply()

    def forward(self, x, y, z):
        q = self.add(x, y)
        p = self.mul(q, z)
        return p


def main():
    # instantiate function object corresponding to `F`.
    func = F()
    # create computational graph of `F`.
    graph = Graph(func)
    graph.compile()

    # initialize input variables.
    x = Tensor(name='Input:x', data=-2.0)
    y = Tensor(name='Input:y', data=5.0)
    z = Tensor(name='Input:z', data=-4.0)

    # perform forward pass of `F` computing its output.
    output = graph.forward(x=x, y=y, z=z)
    # perform backward pass of `F` computing its gradients wrt input variables.
    grads = graph.backward()

    print(f'Output: {output.data}')
    print(f'dF/dx: {grads["Input:x"].data}')
    print(f'dF/dy: {grads["Input:y"].data}')
    print(f'dF/dz: {grads["Input:z"].data}')


if __name__ == '__main__':
    main()
