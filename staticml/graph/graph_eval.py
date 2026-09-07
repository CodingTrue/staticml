from staticml.common.allocator import Allocator
from staticml.graph.lowering import LoweringContext, OperationEntry
from staticml.graph.graph import Graph
from staticml.graph.value import Value
from staticml.buffer import Buffer, BufferView, ASQ
from staticml.program import Program, Kernel


class GraphProgram(Program):
    def __init__(self):
        self.graphs: list[Graph] = []

        self.static_buffer: Buffer = Buffer(name='static_buffer', asq=ASQ.CONSTANT)
        self.dynamic_buffer: Buffer = Buffer(name='dynamic_buffer', asq=ASQ.GLOBAL)

        self.static_allocator: Allocator = Allocator(self.static_buffer)
        self.dynamic_allocator: Allocator = Allocator(self.dynamic_buffer)

        self.dynamic_value_overrides: list[Value] = []
        self.memory_map: dict[Value, BufferView] = {}

        self.operations: list[OperationEntry] = []
        self.values: list[Value] = []

        super().__init__()

    def make_value_dynamic(self, target: Value) -> GraphProgram:
        if target not in self.dynamic_value_overrides:
            self.dynamic_value_overrides.append(target)
        return self

    def add_graph(self, graph: Graph) -> GraphProgram:
        if graph not in self.graphs:
            self.graphs.append(graph)
        return self

    def get_appropriate_allocator(self, target: Value) -> Allocator:
        if target in self.dynamic_value_overrides:
            return self.dynamic_allocator
        return self.static_allocator if target.is_static else self.dynamic_allocator

    def allocate_and_lower(self):
        for graph in self.graphs:
            for node in graph.nodes:
                for value in (*node.inputs, *node.outputs):
                    if value not in self.values:
                        self.values.append(value)

                    allocator = self.get_appropriate_allocator(target=value)

                    if value in self.memory_map: continue
                    self.memory_map[value] = allocator.allocate(size=value.size)

                if len(node.outputs) == 0: continue

                lowering_context = LoweringContext(
                    view_callback=lambda x: self.memory_map.get(x),
                    operations=[]
                )

                node.lower(context=lowering_context)
                self.operations.extend(lowering_context.operations)

    def init_buffers(self):
        self.static_buffer.set_size(size=max(self.static_allocator.max_size, 1)).init()
        self.dynamic_buffer.set_size(size=max(self.dynamic_allocator.max_size, 1)).init()

    def create_kernels(self):
        self.kernels: list[Kernel] = []
        last_launch_config: LaunchConfig = None

        for entry in self.operations:
            if last_launch_config is None or last_launch_config.as_tuple() != entry.launch_config.as_tuple():
                self.kernels.append(Kernel(launch_config=entry.launch_config))

            self.kernels[-1].add_operation(operation=entry.operation)
            last_launch_config = entry.launch_config

    def build(self) -> GraphProgram:
        self.allocate_and_lower()
        self.init_buffers()
        self.create_kernels()

        super().build()
        return self

    def run(self) -> GraphProgram:
        for value in self.values:
            if not value.has_data: continue

            view = self.memory_map[value]
            view.write(data=value._data)

        super().run()
        return self