from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import uuid4
from collections import defaultdict


def _get_inner_ops(obj):
    ops = [v for k, v in vars(obj).items() if isinstance(v, Op)]
    return ops


@dataclass(frozen=True)
class Tensor:
    name: str
    data: Any
    trainable: bool = True

    def update(self, delta):
        data = self.data + delta
        object.__setattr__(self, 'data', data)


@dataclass(frozen=True)
class Node:
    tensor: Any
    grad_fn: Any = None

    @property
    def name(self):
        return self.tensor.name


class Graph:

    def __init__(self, entry):
        self.entry = entry
        self.nodes = None
        self._inputs = None
        self._outputs = None
        self._output = None
        self.reset()

    def reset(self):
        self.nodes = []
        self._inputs = defaultdict(list)
        self._outputs = defaultdict(list)
        self._output = None

    def add_node(self, tensor, grad_fn=None):
        node = Node(tensor=tensor, grad_fn=grad_fn)
        self.nodes.append(node)

    def add_edge(self, src, dst):
        self._inputs[dst.name].append(src.name)
        self._outputs[src.name].append(dst.name)

    def compile(self):
        self.entry.assign_graph(self)

    def forward(self, *args, **kwargs):
        output = self.entry(*args, **kwargs)
        self._output = output
        return output

    def backward(self, grad=None):
        tape = {}
        if self.nodes:
            tail = self.nodes[-1]
            grad = grad or Tensor(name=f'O:{tail.name}', data=1.0)
            tape = {tail.name: grad}
        for node in reversed(self.nodes):
            if node.grad_fn is None or not node.tensor.trainable:
                continue
            grad = node.grad_fn(tape)
            if node.name not in tape:
                tape[node.name] = grad
            else:
                tape[node.name].update(grad.data)
        return tape


class Op(ABC):

    def __init__(self, name=None):
        self.name = name or f'{type(self).__name__}:{str(uuid4())[:8]}'
        self._graph = None
        self._backward_ctx = None

    def __repr__(self):
        rv = f'{type(self).__name__}(name={self.name})'
        return rv

    def __call__(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)
        if self._graph is not None:
            self._extend_graph(output, *args, **kwargs)
            self._set_backward_ctx(output, *args, **kwargs)
        return output

    def _extend_graph(self, *args, **kwargs):
        pass

    def _set_backward_ctx(self, output, *args, **kwargs):
        pass

    def assign_graph(self, graph):
        self._graph = graph
        inner_ops = _get_inner_ops(self)
        for op in inner_ops:
            op.assign_graph(graph)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
