import pyopencl as cl

from dataclasses import dataclass
from enum import Enum

from staticml.device import Device


class ASQ(Enum):
    DYNAMIC = '__global'
    STATIC = '__constant'

class Buffer:
    def __init__(self, name: str, asq: ASQ, type: str, max_size: int = -1):
        self.name = name
        self.asq = asq
        self.type = type
        self.max_size = max_size

        self.cl_buffer: cl.Buffer = None

    def get_cl_buffer(self) -> cl.Buffer:
        if self.cl_buffer is None:
            raise RuntimeError('Buffer was not initialized yet')
        return self.cl_buffer

    def init_cl_buffer(self, size: int | None = None, device: Device | None = None):
        _device = device or Device.get_active()
        _size = size or self.max_size

        if _size > self.max_size:
            raise ValueError(f'Initialization size must not be greate than max size ({self.max_size})')

        if _size <= 0:
            raise ValueError('Buffer must not be initialized with a size less or equal than zero')

        _flags = cl.mem_flags.READ_ONLY if self.asq == ASQ.STATIC else cl.mem_flags.WRITE_ONLY

        self.cl_buffer = cl.Buffer(
            context=_device.get_cl_context(),
            flags=_flags | cl.mem_flags.ALLOC_HOST_PTR,
            size=_size * 4
        )

    def get_type_string(self) -> str:
        return f'{self.asq.value} {self.type}*'

@dataclass
class BufferRange:
    buffer: Buffer
    size: int
    offset: int

class Allocator:
    def __init__(self, buffer: Buffer):
        self.buffer = buffer
        self.sections = []
        self.max_end = 0

    def get_minimum_size(self) -> int:
        return self.max_end + 1

    def allocate(self, size: int) -> BufferRange:
        if size <= 0:
            return

        offset = 0
        index = 0
        last_end = -1

        if (len(self.sections) > 0 and self.sections[-1][1] + size >= self.buffer.max_size) or size > self.buffer.max_size:
            raise RuntimeError(f"Allocator can't allocate more space than the size of the buffer ({self.buffer.max_size})")

        for i, (start, end) in enumerate(self.sections):
            diff = start - last_end - 1

            if diff >= size:
                offset = last_end + 1
                index = i
                break

            offset = end + 1
            index = i + 1
            last_end = end

        self.sections.insert(index, (offset, offset + size - 1))

        _end = self.sections[-1][1]
        self.max_end = _end if _end > self.max_end else self.max_end

        return BufferRange(buffer=self.buffer, size=size, offset=offset)

    def free(self, range: BufferRange):
        _key = (range.offset, range.offset + range.size - 1)
        if _key not in self.sections: return
        self.sections.remove(_key)