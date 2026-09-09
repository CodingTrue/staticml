import numpy as np
from numbers import Number

from typing import Any

from staticml.graph.common_nodes import CommonABNode, MatmulNode, MapNode, TransposeNode
from staticml.graph.graph import Graph
from staticml.graph.value import Value
from staticml.shape import Shape


type TensorArg = Tensor | Number

class Tensor:
    def __init__(
            self,
            data: Any | None = None,
            graph: Graph | None = None,
            value: Value | None = None
    ):
        self.graph = graph or Graph()
        self.value: Value = value or Value(data)

    def __add__(self, other):
        return Tensor._common_new_tensor(left=self, right=other, symbol='+')

    def __sub__(self, other):
        return Tensor._common_new_tensor(left=self, right=other, symbol='-')

    def __mul__(self, other):
        return Tensor._common_new_tensor(left=self, right=other, symbol='*')

    def __truediv__(self, other):
        return Tensor._common_new_tensor(left=self, right=other, symbol='/')

    def __rsub__(self, other):
        return Tensor._common_new_tensor(left=other, right=self, symbol='-')

    def __rtruediv__(self, other):
        return Tensor._common_new_tensor(left=other, right=self, symbol='/')

    def __matmul__(self, other):
        if not Tensor.is_tensor(o=other):
            raise RuntimeError(f"Can't matmul tensor with type {type(other)}")

        sshape, oshape = self.shape.inflated, other.shape.inflated

        if self.shape.x != other.shape.y or ((sshape.z != 1 and oshape.z != 1) and sshape.z != oshape.z):
            raise RuntimeError(f"Can't matmul tensors with mismatched shapes of {self.shape} and {other.shape}")

        node = MatmulNode(a=self.value, b=other.value)

        self.graph.add_node(node=node)
        return Tensor(data=None, graph=self.graph, value=node.result)

    __radd__ = __add__
    __rmul__ = __mul__

    def transpose(self, copy: bool = False) -> Tensor:
        if self.shape.z != -1:
            raise RuntimeError("A transpose on a 3D tensor is not allowed")

        node = TransposeNode(x=self.value, copy=copy)

        self.graph.add_node(node=node)
        return Tensor(data=None, graph=self.graph, value=node.result)

    def map(self, expression: str) -> Tensor:
        node = MapNode(x=self.value, expression=expression)

        self.graph.add_node(node=node)
        return Tensor(data=None, graph=self.graph, value=node.result)

    @property
    def T(self) -> Tensor:
        return self.transpose(copy=False)

    @property
    def data(self) -> np.ndarray:
        return self.value.data

    @property
    def shape(self) -> Shape:
        return self.value.shape

    @property
    def strides(self) -> Shape:
        return self.value.strides

    @property
    def size(self) -> int:
        return self.value.size

    @property
    def has_data(self) -> bool:
        return self.value.has_data

    @staticmethod
    def is_tensor(o: Any) -> bool:
        return isinstance(o, Tensor)

    @staticmethod
    def _common_new_tensor(left: TensorArg, right: TensorArg, symbol: str = '+') -> Tensor:
        left_is_tensor = Tensor.is_tensor(o=left)
        right_is_tensor = Tensor.is_tensor(o=right)

        left_value: Value = left.value if left_is_tensor else Value(left)
        right_value: Value = right.value if right_is_tensor else Value(right)

        if left_is_tensor and right_is_tensor:
            graph = left.graph if left_value.size > right_value.size else right.graph
        elif left_value:
            graph = left.graph
        else:
            graph = right.graph

        node = CommonABNode(a=left_value, b=right_value, symbol=symbol)

        graph.add_node(node=node)
        return Tensor(data=None, graph=graph, value=node.result)