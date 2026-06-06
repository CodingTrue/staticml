from enum import Enum, auto
from typing import Any, Sequence

import numpy as np

from staticml.buffer import BufferRange


class TensorOperation(Enum):
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()

class Tensor:
    def __init__(self, data: Sequence[Any] | None = None, operation: TensorOperation | None = None, children: tuple | None = None):
        self.data = (np.asarray(data) if data is not None else np.empty(0)).astype(np.float32)
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
        return self.buffer_view.size if self.is_allocated() else self.data.size

    def is_static(self) -> bool:
        return self.operation is None

    @staticmethod
    def is_tensor(o: Any) -> bool:
        return isinstance(o, Tensor)

    def __add__(self, other): return Tensor(operation=TensorOperation.ADD, children=(self, other))
    def __sub__(self, other): return Tensor(operation=TensorOperation.SUBTRACT, children=(self, other))
    def __mul__(self, other): return Tensor(operation=TensorOperation.MULTIPLY, children=(self, other))
    def __truediv__(self, other): return Tensor(operation=TensorOperation.DIVIDE, children=(self, other))

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__