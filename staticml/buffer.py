from dataclasses import dataclass
from enum import Enum


class ASQ(Enum):
    DYNAMIC = '__global'
    STATIC = '__constant'

@dataclass
class Buffer:
    def __init__(self, name: str, asq: ASQ, type: str):
        self.name = name
        self.asq = asq
        self.type = type

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
        offset = 0
        index = 0
        last_end = -1

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