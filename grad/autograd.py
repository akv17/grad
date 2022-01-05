import pprint
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import uuid4
from collections import defaultdict


def get_inner_ops(obj):
    ops = [v for k, v in vars(obj).items() if isinstance(v, Op)]
    return ops


@dataclass(frozen=True)
class Tensor:
    name: str
    data: Any


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

    def backward(self):
        tape = {}
        if self.nodes:
            tail = self.nodes[-1]
            tape = {tail.name: Tensor(name=tail.name, data=1.0)}
        for node in reversed(self.nodes):
            if node.grad_fn is None:
                continue
            grad = node.grad_fn(tape)
            if node.name not in tape:
                tape[node.name] = grad
            else:
                tape[node.name] += grad
        return tape


class Op(ABC):

    def __init__(self):
        self.name = f'{type(self).__name__}:{str(uuid4())[:8]}'
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
        inner_ops = get_inner_ops(self)
        for op in inner_ops:
            op.assign_graph(graph)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class Add(Op):

    def _extend_graph(self, output, a, b):
        self._graph.add_node(a, grad_fn=self._grad_a)
        self._graph.add_node(b, grad_fn=self._grad_b)
        self._graph.add_node(output)
        self._graph.add_edge(a, output)
        self._graph.add_edge(b, output)

    def _grad_a(self, tape):
        upstream = tape[self.name]
        local = 1.0
        downstream = upstream.data * local
        grad = Tensor(name=self.name, data=downstream)
        return grad

    def _grad_b(self, tape):
        upstream = tape[self.name]
        local = 1.0
        downstream = upstream.data * local
        grad = Tensor(name=self.name, data=downstream)
        return grad

    def forward(self, a, b):
        output = a.data + b.data
        output = Tensor(name=self.name, data=output)
        return output


class Mul(Op):

    def _extend_graph(self, output, a, b):
        self._graph.add_node(a, grad_fn=self._grad_a)
        self._graph.add_node(b, grad_fn=self._grad_b)
        self._graph.add_node(output)
        self._graph.add_edge(a, output)
        self._graph.add_edge(b, output)

    def _set_backward_ctx(self, output, a, b):  # noqa
        self._backward_ctx = {'a': a, 'b': b}

    def _grad_a(self, tape):
        upstream = tape[self.name]
        local = self._backward_ctx['b']
        downstream = upstream.data * local.data
        grad = Tensor(name=self.name, data=downstream)
        return grad

    def _grad_b(self, tape):
        upstream = tape[self.name]
        local = self._backward_ctx['a']
        downstream = upstream.data * local.data
        grad = Tensor(name=self.name, data=downstream)
        return grad

    def forward(self, a, b):
        output = a.data * b.data
        output = Tensor(name=self.name, data=output)
        return output
