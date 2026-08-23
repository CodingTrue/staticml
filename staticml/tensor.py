from dataclasses import dataclass

import numpy as np

from enum import Enum
from typing import Any, Iterable

from staticml.dtype import float32


type TensorArg = Tensor | Number


class TensorOperation(Enum):
    ADD = '_add'
    SUB = '_sub'
    MUL = '_mul'
    DIV = '_div'
    MATMUL = '_matmul'

@dataclass
class TensorShape:
    x: int = 1
    y: int = 1
    z: int = 1

    def as_tuple(self) -> tuple[int, int, int]:
        return self.x, self.y, self.z

def _broadcast_shapes(left: TensorShape, right: TensorShape) -> TensorShape:
    for l_dim, r_dim in zip(left.as_tuple(), right.as_tuple()):
        if l_dim == r_dim or (l_dim == 1 or r_dim == 1): continue
        raise ValueError(f"Operands could not be broadcasted together with shapes {left} and {right}")

    return TensorShape(
        x=right.x if left.x == 1 else left.x,
        y=right.y if left.y == 1 else left.y,
        z=right.z if left.z == 1 else left.z,
    )

class Tensor:
    def __init__(
            self,
            data: Any | None = None,
            args: tuple[TensorArg] | None = None,
            shape: TensorShape | None = None,
            strides: TensorShape | None = None
    ):
        self._data: np.ndarray = None

        self.args: tuple = args or tuple()
        self._shape: TensorShape = None
        self._strides: TensorShape = None

        if strides is None and isinstance(shape, TensorShape):
            strides = TensorShape(x=1, y=shape.x, z=shape.x * shape.y)

        self.set_data(data=np.asarray(data, dtype=float32.dtype), shape=shape, strides=strides)

    def __add__(self, other):
        return Tensor._common_new_tensor(self, other, TensorOperation.ADD)

    def __sub__(self, other):
        return Tensor._common_new_tensor(self, other, TensorOperation.SUB)

    def __mul__(self, other):
        return Tensor._common_new_tensor(self, other, TensorOperation.MUL)

    def __truediv__(self, other):
        return Tensor._common_new_tensor(self, other, TensorOperation.DIV)

    def __rsub__(self, other):
        return Tensor._common_new_tensor(other, self, TensorOperation.SUB)

    def __rtruediv__(self, other):
        return Tensor._common_new_tensor(other, self, TensorOperation.DIV)

    def __matmul__(self, other):
        if not Tensor.is_tensor(o=other):
            raise RuntimeError(f"Can't matmul tensor with type {type(other)}")

        if self.shape.x != other.shape.y or self.shape.z != other.shape.z:
            raise RuntimeError(f"Can't matmul tensors with mismatched shapes of {self.shape} and {other.shape}")

        shape = TensorShape(
            x=other.shape.x,
            y=self.shape.y,
        )
        return Tensor(data=None, args=(TensorOperation.MATMUL, self, other), shape=shape)

    __radd__ = __add__
    __rmul__ = __mul__

    def set_data(self, data: np.ndarray, shape: TensorShape | None, strides: TensorShape | None):
        self._data = data

        item_size = self._data.dtype.itemsize

        self.shape = shape or TensorShape(*self._data.shape[::-1])
        self.strides = strides or TensorShape(*tuple(x // item_size for x in self._data.strides[::-1]))

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: Tensor | np.ndarray):
        if Tensor.is_tensor(o=value):
            self.set_data(data=value.data, shape=x.shape)
        elif isinstance(value, np.ndarray):
            self.set_data(data=value)
        else:
            raise ValueError(f"Tensor data can't be set to type of {type(value)}")

    @property
    def shape(self) -> TensorShape:
        return self._shape

    @shape.setter
    def shape(self, value) -> TensorShape:
        if isinstance(value, TensorShape):
            self._shape = value
        elif isinstance(value, tuple[int, ...]):
            self._shape = TensorShape(*value)
        else:
            raise ValueError(f"Tensor shape can't be set to type of {type(value)}")

        if self.has_data:
            self._data.shape = self._shape.as_tuple()[::-1]

    @property
    def strides(self) -> TensorShape:
        return self._strides

    @strides.setter
    def strides(self, value) -> TensorShape:
        if isinstance(value, TensorShape):
            self._strides = value
        elif isinstance(value, tuple):
            self._strides = TensorShape(*value)
        else:
            raise ValueError(f"Tensor strides can't be set to type of {type(value)}")

        if self.has_data:
            item_size = self._data.dtype.itemsize
            self._data.strides = tuple(x * item_size for x in self._strides.as_tuple()[::-1])

    @property
    def size(self) -> int:
        return self._data.size if self.has_data else 0

    @property
    def has_data(self) -> bool:
        return self._data.ndim != 0

    @property
    def is_static(self) -> bool:
        return len(self.args) == 0

    @staticmethod
    def is_tensor(o: Any) -> bool:
        return isinstance(o, Tensor)

    @staticmethod
    def _common_new_tensor(left, right, op: TensorOperation) -> Tensor:
        la, ra = (left, right) if Tensor.is_tensor(o=left) else (right, left)
        shape = _broadcast_shapes(left=left.shape, right=right.shape) if Tensor.is_tensor(o=ra) else la.shape

        return Tensor(data=None, args=(op, left, right), shape=shape)