from staticml.buffer import Buffer, BufferView


class Allocator:
    def __init__(self, buffer: Buffer):
        self.buffer = buffer

        self.views: list[BufferView] = []

    def get_max_size(self) -> int:
        if len(self.views) == 0: return 0

        view = self.views[-1]
        return view.offset + view.size

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
        return view

    def free(self, view: BufferView):
        if not view in self.views:
            raise RuntimeError(f"Allocator has not generated the view '{view}'")

        self.views.remove(view)