from dataclasses import dataclass

import numpy as np
import pyopencl as cl

from enum import Enum

from staticml.device import Device
from staticml.dtype import DType, uint32

BASE_DTYPE: DType = uint32

class ASQ(Enum):
    GLOBAL = '__global'
    CONSTANT = '__constant'

@dataclass
class BufferView:
    buffer: Buffer
    size: int
    offset: int = 0

    def write(self, data, offset: int = 0) -> BufferView:
        if len(data) + offset + self.offset > self.size:
            raise RuntimeError('View is writing to indices exceeding the allowed range')
        return self

    def read(self, size: int, offset: int = 0) -> np.ndarray:
        if size + offset > self.size:
            raise RuntimeError('View is reading from indices exceeding the allowed range')
        return self.buffer.read(size=size, offset=offset + self.offset)

class Buffer:
    def __init__(self, name: str = '', asq: ASQ = ASQ.GLOBAL):
        self.name = name or f'buffer_{hash(self):0x}'
        self.asq = asq
        self._size = 0

        self.flags = 0

        if self.asq == ASQ.GLOBAL:
            self.flags = cl.mem_flags.READ_WRITE
        elif self.asq == ASQ.CONSTANT:
            self.flags = cl.mem_flags.READ_ONLY

        self._buffer: cl.Buffer = None
        self._device: Device = None

    def __repr__(self):
        return f"Buffer('{self._name}')"

    def _check_init(self, state: bool):
        state_check = state == self.is_initialized

        if state_check:
            return

        if state:
            raise RuntimeError('Buffer is not initialized')
        else:
            raise RuntimeError('Buffer is already initialized')

    def expand(self, size: int) -> Buffer:
        self._check_init(state=False)

        if size < 0:
            raise ValueError(f"Size may not be smaller than 0")

        self._size += size
        return self

    def set_size(self, size: int) -> Buffer:
        self._check_init(state=False)

        if size <= 0:
            raise ValueError(f"Size may not be smaller than 1")

        self._size = size
        return self

    def init(self, device: Device | None = None) -> Buffer:
        self._check_init(state=False)

        if self._size == 0:
            raise RuntimeError(f"Can't initialized buffer of size 0")

        device = device or Device.active()

        self._buffer = cl.Buffer(
            context=device.context,
            flags=self.flags,
            size=self._size * BASE_DTYPE.dtype.itemsize,
            hostbuf=None
        )

        self._device = device
        return self

    def write(self, data, offset: int = 0) -> Buffer:
        self._check_init(state=True)

        _data = np.asarray(data, dtype=BASE_DTYPE.dtype)

        if _data.size + offset > self._size:
            raise ValueError(
                f"Can't write data (size = {_data.size}) to buffer (size = {self._size}) with offset of {offset}")

        cl.enqueue_copy(
            queue=self._device.queue,
            dest=self._buffer,
            src=_data,
            dst_offset=offset * BASE_DTYPE.dtype.itemsize
        )

        return self

    def read(self, size: int, offset: int = 0) -> np.ndarray:
        self._check_init(state=True)

        if size + offset > self._size:
            raise ValueError(
                f"Can't read data (size = {size}) from buffer (size = {self._size}) with offset of {offset}")

        _data = np.empty(size, dtype=BASE_DTYPE.dtype)

        cl.enqueue_copy(
            queue=self._device.queue,
            dest=_data,
            src=self._buffer,
            src_offset=offset * BASE_DTYPE.dtype.itemsize
        )

        return _data

    def view(self, size: int, offset: int = 0) -> BufferView:
        if size + offset > self._size:
            raise ValueError("Can't return view that has a bigger range than the buffer itself")

        return BufferView(
            buffer=self,
            size=size,
            offset=offset
        )

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_initialized(self) -> bool:
        return self._buffer is not None

    @property
    def buffer(self) -> cl.Buffer:
        self._check_init(state=True)
        return self._buffer

    @staticmethod
    def is_buffer(o: Any) -> bool:
        return isinstance(o, Buffer)