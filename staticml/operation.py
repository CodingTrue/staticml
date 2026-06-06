from numbers import Number

from staticml.buffer import Allocator
from staticml.tensor import Tensor


class Operation:
    def __init__(self, arguments: list = None):
        self.arguments = arguments or []

    def allocate(self, allocator: Allocator):
        return