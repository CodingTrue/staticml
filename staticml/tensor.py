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

@dataclass
class TensorShape:
    x: int = 1
    y: int = 1
    z: int = 1

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
        return Tensor(data=None, args=(TensorOperation.ADD, self, other))

    def __sub__(self, other):
        return Tensor(data=None, args=(TensorOperation.SUB, self, other))

    def __mul__(self, other):
        return Tensor(data=None, args=(TensorOperation.MUL, self, other))

    def __truediv__(self, other):
        return Tensor(data=None, args=(TensorOperation.DIV, self, other))

    def __rsub__(self, other):
        return Tensor(data=None, args=(TensorOperation.SUB, other, self))

    def __rtruediv__(self, other):
        return Tensor(data=None, args=(TensorOperation.DIV, other, self))

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