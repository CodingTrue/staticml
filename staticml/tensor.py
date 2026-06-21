from enum import Enum, auto
from numbers import Number
from typing import Any, Sequence

import numpy as np

from staticml.buffer import BufferRange


class TensorOperation(Enum):
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MATMUL = auto()

class Tensor:
    def __init__(self,
                 data: Sequence[Any] | None = None,
                 operation: TensorOperation | None = None,
                 children: tuple | None = None,
                 shape: tuple[int, ...] | None = None
    ):
        self._data = (np.asarray(data) if data is not None else np.empty(0)).astype(np.float32)
        self._shape = shape or self.data.shape
        self.operation = operation
        self.children = children
        self.buffer_view: BufferRange = None

        if not self.is_static() and self.children is None:
            raise ValueError("Children can't be be empty when tensor is non-static")

    def set_buffer_view(self, view: BufferRange):
        self.buffer_view = view

    def is_allocated(self) -> bool:
        return self.buffer_view is not None

    def get_size(self) -> int:
        return self.buffer_view.size if self.is_allocated() else self._data.size

    def is_static(self) -> bool:
        return self.operation is None

    def get_shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Can't set tensor data to type '{type(value)}'")
        self._data = value
        self._shape = self._data.shape

    @staticmethod
    def is_tensor(o: Any) -> bool:
        return isinstance(o, Tensor)

    def __add__(self, other):
        return Tensor(operation=TensorOperation.ADD, children=(self, other), shape=_get_common_shape(self, other))
    def __sub__(self, other):
        return Tensor(operation=TensorOperation.SUBTRACT, children=(self, other), shape=_get_common_shape(self, other))
    def __mul__(self, other):
        return Tensor(operation=TensorOperation.MULTIPLY, children=(self, other), shape=_get_common_shape(self, other))
    def __truediv__(self, other):
        return Tensor(operation=TensorOperation.DIVIDE, children=(self, other), shape=_get_common_shape(self, other))
    def __matmul__(self, other):
        if not Tensor.is_tensor(other):
            raise RuntimeError(f"Can't matmul a tensor with type {type(other)}")

        shape_a = self.get_shape()[::-1]
        shape_b = other.get_shape()[::-1]

        if shape_b[1] != shape_a[0]:
            raise RuntimeError(f"Can't matmul tensors with dimension {shape_b}")

        return Tensor(operation=TensorOperation.MATMUL, children=(self, other), shape=(shape_a[1], shape_b[0]))

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__

def _get_common_shape(a: Tensor | Number, b: Tensor | Number) -> tuple[int, ...]:
    a_is_tensor = Tensor.is_tensor(a)
    b_is_tensor = Tensor.is_tensor(b)

    if a_is_tensor and not b_is_tensor:
        return a.get_shape()

    if b_is_tensor and not a_is_tensor:
        return b.get_shape()

    a_shape = a.get_shape()
    b_shape = b.get_shape()

    if len(a_shape) != len(b_shape):
        raise RuntimeError(f"Can't find common shape for mismatched dimensions {a_shape} and {b_shape}")

    return min(a_shape, b_shape)