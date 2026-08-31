from dataclasses import dataclass

from staticml.buffer import Buffer, ASQ, BufferView
from staticml.common import Allocator, lower_tensor, LoweringContext
from staticml.device import Device
from staticml.operation import Operation
from staticml.program import Program, Kernel, LaunchConfig
from staticml.tensor import Tensor


@dataclass(frozen=True)
class OperationEntry:
    operation: Operation
    launch_config: LaunchConfig

    def is_equal(self, other: OperationEntry) -> bool:
        return self.launch_config.as_tuple() == other.launch_config.as_tuple()

class TensorEvaluationProgram(Program):
    def __init__(self, tensor: Tensor):
        self.tensor_map: dict[Tensor, BufferView] = {}
        self.tensor_lifetimes: dict[Tensor, list[Tensor]] = {}
        self.operations: list[OperationEntry] = []

        self.static_tensors: list[Tensor] = []
        self.dynamic_tensors: list[Tensor] = []

        self.static_buffer = Buffer(name='static_buffer', asq=ASQ.CONSTANT)
        self.dynamic_buffer = Buffer(name='dynamic_buffer', asq=ASQ.GLOBAL)

        self.static_allocator = Allocator(buffer=self.static_buffer)
        self.dynamic_allocator = Allocator(buffer=self.dynamic_buffer)

        self.visited_tensors: set[Tensor] = set()
        self.kernels: list[Kernel] = []

        self.build_graph(tensor=tensor)
        self.allocate_tensors()
        self.generate_kernels()

        super().__init__(kernels=self.kernels)

    def build(self, device: Device | None = None) -> TensorEvaluationProgram:
        device = device or Device.active()

        if not self.operations:
            return self

        self.static_buffer.set_size(size=self.static_allocator.get_max_size()).init(device=device)
        self.dynamic_buffer.set_size(size=self.dynamic_allocator.get_max_size()).init(device=device)

        return super().build(device)

    def run(self) -> TensorEvaluationProgram:
        if not self.operations:
            return self

        for tensor in self.static_tensors:
            view = self.get_tensor_view(tensor=tensor)
            view.write(data=tensor._data)

        super().run()
        return self

    def get_tensor_view(self, tensor: Tensor) -> BufferView:
        if not tensor in self.tensor_map:
            raise RuntimeError('Tensor not in tensor map')
        return self.tensor_map[tensor]

    def generate_kernels(self):
        self.kernels: list[Kernel] = []
        last_entry: OperationEntry = None

        for entry in self.operations:
            if not last_entry or not last_entry.is_equal(entry):
                knl = Kernel(launch_config=entry.launch_config)
                self.kernels.append(knl)

            self.kernels[-1].add_operation(operation=entry.operation)
            last_entry = entry

    def build_graph(self, tensor: Tensor):
        if tensor in self.visited_tensors: return
        self.visited_tensors.add(tensor)

        if tensor.is_static:
            self.static_tensors.append(tensor)
            return

        self.tensor_lifetimes[tensor] = []

        for arg in tensor.args:
            if not Tensor.is_tensor(o=arg): continue
            self.build_graph(tensor=arg)

            if arg.is_static: continue
            self.tensor_lifetimes[tensor].append(arg)

        self.dynamic_tensors.append(tensor)

    def allocate_tensors(self):
        for tensor in self.static_tensors:
            self.tensor_map[tensor] = self.static_allocator.allocate(size=tensor.size)

        for tensor in self.dynamic_tensors:
            lc: LoweringContext = lower_tensor(
                tensor=tensor,
                view_callback=lambda t: self.get_tensor_view(tensor=t),
                allocator=self.dynamic_allocator
            )

            self.tensor_map[tensor] = lc.out_view
            self.operations.append(OperationEntry(
                operation=lc.out_operation,
                launch_config=lc.out_launch_config
            ))

            deaths = self.tensor_lifetimes[tensor]
            for death in deaths:
                self.dynamic_allocator.free(view=self.get_tensor_view(tensor=death))