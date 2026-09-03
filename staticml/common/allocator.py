from staticml.buffer import Buffer, BufferView


class Allocator:
    def __init__(self, buffer: Buffer):
        self.buffer = buffer

        self.views: list[BufferView] = []
        self.max_size = 0

    def allocate(self, size: int = 0):
        last_end = 0
        index = 0

        for i, view in enumerate(self.views):
            span = view.offset - last_end
            index = i

            if span >= size:
                break
            else:
                index += 1
            last_end = view.offset + view.size

        view = self.buffer.view(size=size, offset=last_end, unsafe_view=True)
        self.views.insert(index, view)

        last_view = self.views[-1]
        self.max_size = max(self.max_size, last_view.size + last_view.offset)

        return view

    def free(self, view: BufferView):
        if not view in self.views:
            raise RuntimeError(f"Allocator has not generated the view '{view}'")

        self.views.remove(view)