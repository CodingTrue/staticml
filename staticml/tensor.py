from enum import Enum, auto
from typing import Any


class TensorOperation(Enum):
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()

class Tensor:
    def __init__(self, operation: TensorOperation | None = None, children: tuple | None = None):
        self.operation = operation
        self.children = children

        if not self.is_static() and self.children is None:
            raise ValueError("Children can't be be empty when tensor is non-static")

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