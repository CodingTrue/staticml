import pyopencl as cl
import numpy as np

from numbers import Number

from staticml.buffer import Allocator, Buffer, ASQ
from staticml.device import Device
from staticml.graph import ComputeGraph
from staticml.kernel import Kernel
from staticml.lowering import lower_tensor
from staticml.operation import Operation, AXBZOperation
from staticml.tensor import Tensor, TensorOperation



class Program:
    def __init__(self, graph: ComputeGraph, device: Device | None = None):
        if len(graph.dynamic_tensors) == 0:
            raise RuntimeError('Creating a program with an empty compute graph leads to ambiguity for return values')

        self._device = device or Device.get_active()
        self.graph: ComputeGraph = graph

        _max_alloc = self._device.get_max_allocation_size()
        _db = Buffer(name='dynamic_buffer', asq=ASQ.DYNAMIC, type='float', max_size=_max_alloc)
        _sb = Buffer(name='static_buffer', asq=ASQ.STATIC, type='float', max_size=_max_alloc)

        self.dynamic_allocator = Allocator(buffer=_db)
        self.static_allocator = Allocator(buffer=_sb)

        self.kernel: Kernel = Kernel(buffers=[self.dynamic_allocator.buffer, self.static_allocator.buffer])

    def init_buffers(self):
        self.dynamic_allocator.buffer.init_cl_buffer(
            size=self.dynamic_allocator.get_minimum_size(),
            device=self._device
        )

        self.static_allocator.buffer.init_cl_buffer(
            size=self.static_allocator.get_minimum_size(),
            device=self._device
        )

    def allocate_tensors(self):
        for tensor in self.graph.static_tensors:
            tensor.set_buffer_view(view=self.static_allocator.allocate(size=tensor.get_size()))

        for tensor in self.graph.dynamic_tensors:
            _op = lower_tensor(tensor=tensor)
            _op.allocate(allocator=self.dynamic_allocator)

            self.kernel.add_operation(operation=_op)

            deaths = self.graph.liftetimes.get(tensor)
            if deaths is None: continue

            for death in deaths:
                if death.is_static(): continue
                self.dynamic_allocator.free(range=death.buffer_view)

    def build(self) -> Kernel:
        self.allocate_tensors()
        self.init_buffers()
        self.kernel.compile()

        return self

    def run(self) -> Tensor:
        for tensor in self.graph.static_tensors:
            cl.enqueue_copy(
                queue=self._device.get_cl_queue(),
                dest=self.static_allocator.buffer.get_cl_buffer(),
                src=tensor.data, dst_offset=tensor.buffer_view.offset * 4
            )

        cl.enqueue_nd_range_kernel(
            queue=self._device.get_cl_queue(),
            kernel=self.kernel.get_cl_kernel(),
            global_work_size=self.kernel.work_size,
            local_work_size=None
        )

        _last_tensor = self.graph.dynamic_tensors[-1]
        dynamic_data = np.empty(self.kernel.work_size[::-1], dtype=np.float32)
        cl.enqueue_copy(
            queue=self._device.get_cl_queue(),
            dest=dynamic_data, src=self.dynamic_allocator.buffer.get_cl_buffer(),
            src_offset=_last_tensor.buffer_view.offset * 4
        )

        _last_tensor.data = dynamic_data.reshape(_last_tensor.get_shape())
        return _last_tensor