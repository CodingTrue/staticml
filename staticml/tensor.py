from dataclasses import dataclass

import numpy as np

from enum import Enum
from typing import Any, Iterable


type TensorArg = Tensor | Number

def _get_common_size(args: Iterable[TensorArg]) -> int:
    return max(arg.size for arg in args if Tensor.is_tensor(o=arg))

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
            shape: TensorShape | None = None
    ):
        self._data = np.asarray(data, dtype=np.float32)
        self._has_data = data is not None

        self._args = args or tuple()
        self._shape = shape or TensorShape(x=_get_common_size(args=self._args) if not self._has_data else 0)
        self._is_static = len(self._args) == 0

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

    @property
    def size(self) -> int:
        return self._data.size if self._has_data else 0

    @property
    def has_data(self) -> bool:
        return self._has_data

    @property
    def shape(self) -> TensorShape:
        return TensorShape(*self._data.shape[::-1]) if self._has_data else self._shape

    @property
    def args(self) -> tuple[Any]:
        return self._args

    @property
    def is_static(self) -> bool:
        return self._is_static

    @staticmethod
    def is_tensor(o: Any) -> bool:
        return isinstance(o, Tensor)

    @staticmethod
    def _common_new_tensor(left, right, op: TensorOperation) -> Tensor:
        la, ra = (left, right) if Tensor.is_tensor(o=left) else (right, left)
        shape = _broadcast_shapes(left=left.shape, right=right.shape) if Tensor.is_tensor(o=ra) else la.shape

        return Tensor(data=None, args=(op, left, right), shape=shape)