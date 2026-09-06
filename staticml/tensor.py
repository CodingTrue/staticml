from dataclasses import dataclass

import numpy as np

from enum import Enum
from typing import Any, Iterable

from staticml.dtype import float32
from staticml.shape import Shape

type TensorArg = Tensor | Number


class TensorOperation(Enum):
    ADD = '_add'
    SUB = '_sub'
    MUL = '_mul'
    DIV = '_div'
    MATMUL = '_matmul'
    TRANSPOSE = '_transpose'
    MAP = '_map'

def _broadcast_shapes(left: Shape, right: Shape) -> Shape:
    for l_dim, r_dim in zip(left.as_tuple(), right.as_tuple()):
        if l_dim == r_dim or (l_dim == 1 or r_dim == 1): continue
        raise ValueError(f"Operands could not be broadcasted together with shapes {left} and {right}")

    return Shape(
        x=max(left.x, right.x),
        y=max(left.y, right.y),
        z=max(left.z, right.z),
    )

class Tensor:
    def __init__(
            self,
            data: Any | None = None,
            args: tuple[TensorArg] | None = None,
            shape: Shape | None = None,
            strides: Shape | None = None
    ):
        self._data: np.ndarray = None

        self.args: tuple = args or tuple()
        self._shape: Shape = None
        self._strides: Shape = None

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

        sshape, oshape = self.shape.inflated, other.shape.inflated

        if self.shape.x != other.shape.y or ((sshape.z != 1 and oshape.z != 1) and sshape.z != oshape.z):
            raise RuntimeError(f"Can't matmul tensors with mismatched shapes of {self.shape} and {other.shape}")

        shape = Shape(
            x=other.shape.x,
            y=self.shape.y,
            z=max(self.shape.z, other.shape.z)
        )

        strides = Shape(
            x=1 if shape.x != -1 else -1,
            y=shape.x if shape.y != -1 else -1,
            z=shape.x * shape.y if shape.z != -1 else -1,
        )

        return Tensor(data=None, args=(TensorOperation.MATMUL, self, other), shape=shape, strides=strides)

    __radd__ = __add__
    __rmul__ = __mul__

    def set_data(self, data: np.ndarray, shape: Shape | None = None, strides: Shape | None = None):
        self._data = data

        item_size = self._data.dtype.itemsize

        new_shape = shape or Shape(*self._data.shape[::-1])
        new_strides = strides or Shape(*tuple(x // item_size for x in self._data.strides[::-1]))

        if self.has_data:
            self._data = np.lib.stride_tricks.as_strided(
                self._data,
                shape=new_shape.as_tuple(reverse=True),
                strides=tuple(x * item_size for x in new_strides.as_tuple(reverse=True))
            )

        self._shape = new_shape
        self._strides = new_strides

    def transpose(self, copy: bool = False) -> Tensor:
        if self._shape.z != -1:
            raise RuntimeError("A transpose on a 3D tensor is not allowed")

        return Tensor(
            data=self._data.copy() if copy else self._data.view(),
            args=(TensorOperation.TRANSPOSE, self),
            shape=Shape.from_numpy_shape(self._shape.as_tuple()),
            strides=Shape.from_numpy_shape(self._strides.as_tuple())
        )

    def map(self, expression: str) -> Tensor:
        return Tensor(
            data=None,
            args=(TensorOperation.MAP, self, expression),
            shape=Shape(*self._shape.as_tuple()),
            strides=Shape(*self._strides.as_tuple())
        )

    @property
    def T(self) -> Tensor:
        return self.transpose(copy=False)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: Tensor | np.ndarray):
        if Tensor.is_tensor(o=value):
            self.set_data(data=value.data, shape=value.shape, strides=value.strides)
        elif isinstance(value, np.ndarray):
            self.set_data(data=value)
        else:
            raise ValueError(f"Tensor data can't be set to type of {type(value)}")

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def strides(self) -> Shape:
        return self._strides

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

        if Tensor.is_tensor(o=la) and Tensor.is_tensor(o=ra):
            ldim = la.shape.inflated.as_tuple()
            rdim = ra.shape.inflated.as_tuple()
            source_tensor = la if all(l >= r for l, r in zip(ldim, rdim)) else ra
        else:
            source_tensor = la

        strides = source_tensor.strides

        return Tensor(data=None, args=(op, left, right), shape=shape, strides=strides)