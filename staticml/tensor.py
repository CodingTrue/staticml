import numpy as np

from typing import Any


class Tensor:
    def __init__(self, data: Any | None = None):
        self._data = np.asarray(data, dtype=np.float32)
        self._has_data = data is not None

    @property
    def size(self) -> int:
        return self._data.size if self._has_data else 0

    @property
    def has_data(self) -> bool:
        return self._has_data