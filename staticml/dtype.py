from typing import Any

import numpy as np


class DType:
    def __init__(self, dtype: np.dtype, c_name: str):
        self._dtype = dtype
        self._c_name = c_name

    def cast(self, o: Any) -> Any:
        return self.dtype.type(o)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def c_name(self) -> str:
        return self._c_name

uint32 = DType(dtype=np.dtype(np.uint32), c_name='uint')
int32 = DType(dtype=np.dtype(np.int32), c_name='int')
float32 = DType(dtype=np.dtype(np.float32), c_name='float')