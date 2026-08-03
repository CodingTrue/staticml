from staticml.buffer import Buffer

class Allocator:
    def __init__(self, buffer: Buffer):
        self.buffer = buffer
        self.pointer = 0

    def allocate(self, size: int = 0):
        self.buffer.expand(size=size)
        view = self.buffer.view(size=size, offset=self.pointer)

        self.pointer += size
        return view