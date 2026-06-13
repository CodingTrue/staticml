from dataclasses import dataclass
from enum import Enum


class ASQ(Enum):
    DYNAMIC = '__global'
    STATIC = '__constant'

class Buffer:
    def __init__(self, name: str, asq: ASQ, type: str, max_size: int = -1):
        self.name = name
        self.asq = asq
        self.type = type
        self.max_size = max_size

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
        return BufferRange(buffer=self.buffer, size=size, offset=offset)

    def free(self, range: BufferRange):
        _key = (range.offset, range.offset + range.size - 1)
        if _key not in self.sections: return
        self.sections.remove(_key)