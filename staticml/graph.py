from staticml.tensor import Tensor

def _add_if_missing(e, l):
    if e in l: return
    l.append(e)

class ComputeGraph:
    def __init__(self, root: Tensor):
        self.dynamic_tensors: list[Tensor] = []
        self.static_tensors: list[Tensor] = []
        self.keepers: dict[Tensor, Tensor] = {}
        self.liftetimes: dict[Tensor, list[Tensor]] = {}

        self._search(tensor=root)

        for child, keeper in self.keepers.items():
            if not keeper in self.liftetimes:
                self.liftetimes[keeper] = []

            self.liftetimes[keeper].append(child)

    def _search(self, tensor: Tensor):
        if tensor.is_static():
            _add_if_missing(tensor, self.static_tensors)
            return

        for child in tensor.children:
            if not Tensor.is_tensor(child): continue
            self.keepers[child] = tensor

            self._search(tensor=child)

        _add_if_missing(tensor, self.dynamic_tensors)