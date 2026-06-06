from abc import ABC, abstractmethod
from numbers import Number

from staticml.buffer import Allocator
from staticml.tensor import Tensor


class Operation(ABC):
    def __init__(self, arguments: list = None):
        self.arguments = arguments or []

    @abstractmethod
    def allocate(self, allocator: Allocator):
        return