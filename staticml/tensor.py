import numpy as np

from enum import Enum
from typing import Any


class TensorOperation(Enum):
    ADD = '_add'
    SUB = '_sub'
    MUL = '_mul'
    DIV = '_div'

class Tensor:
    def __init__(self, data: Any | None = None, args: tuple[Any] | None = None):
        self._data = np.asarray(data, dtype=np.float32)
        self._has_data = data is not None

        self._args = args
        self._is_static = self._args is not None

    def __add__(self, other):
        return Tensor(data=None, args=(TensorOperation.ADD, self, other))

    def __sub__(self, other):
        return Tensor(data=None, args=(TensorOperation.SUB, self, other))

    def __mul__(self, other):
        return Tensor(data=None, args=(TensorOperation.MUL, self, other))

    def __truediv__(self, other):
        return Tensor(data=None, args=(TensorOperation.DIV, self, other))

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__

    @property
    def size(self) -> int:
        return self._data.size if self._has_data else 0

    @property
    def has_data(self) -> bool:
        return self._has_data

    @property
    def args(self) -> tuple[Any]:
        return self._args

    @property
    def is_static(self) -> bool:
        return self._is_static

    @staticmethod
    def is_tensor(o: Any) -> bool:
        return isinstance(o, Tensor)