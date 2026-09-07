import numpy as np

from numbers import Number

from staticml.dtype import float32
from staticml.shape import Shape


def _shape_produt(shape: Shape):
    inflated = shape.inflated
    return inflated.x * inflated.y * inflated.z

class Value:
    def __init__(
            self,
            data: Number | np.ndarray | None = None,
            shape: Shape | None = None,
            strides: Shape | None = None
    ):
        if isinstance(data, Number):
            data = [data]

        self._data: np.ndarray = np.asarray(data, dtype=float32.dtype)

        if self.has_data:
            itemsize = self._data.dtype.itemsize

            self.shape = Shape.from_numpy_shape(self._data.shape)
            self.strides = Shape.from_numpy_shape(tuple(x // itemsize for x in self._data.strides))
        else:
            self.shape = shape
            self.strides = strides

            if self.shape is None:
                raise ValueError("Expected explicit shape when data is not given")

            if self.strides is None:
                raise ValueError("Expected explicit strides when data is not given")

    @property
    def size(self) -> int:
        if self.has_data:
            return self._data.size
        return _shape_produt(shape=self.shape)

    @property
    def is_static(self) -> bool:
        return self.has_data

    @property
    def has_data(self) -> bool:
        return self._data.ndim != 0

    @staticmethod
    def empty() -> Value:
        return Value(shape=Shape(), strides=Shape())

    def __repr__(self):
        return f'Value_{id(self):0x}'