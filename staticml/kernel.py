import pyopencl as cl

from textwrap import indent

from staticml.buffer import Buffer
from staticml.device import Device
from staticml.operation import Operation

class Kernel:
    def __init__(self, buffers: list[Buffer] | None = None, operations: list[Operation] | None = None):
        self.buffers: list[Buffer] = buffers or []
        self.operations: list[Operation] = operations or []
        self.cl_kernel: cl.Kernel = None
        self.work_size = None

    def get_cl_kernel(self) -> cl.Kernel:
        if self.cl_kernel is None:
            raise RuntimeError('Kernel is not compiled yet')
        return self.cl_kernel

    def add_operation(self, operation: Operation):
        _work_size = operation.get_work_size()

        if self.work_size is None:
            self.work_size = _work_size

        if _work_size > self.work_size:
            raise ValueError(f"Operation work-size '{_work_size}' exceeds epxected work-size of '{self.work_size}'")

        self.operations.append(operation)

    def get_definition(self) -> str:
        call_strings = []
        for op in self.operations:
            call_strings.append(op.get_call_string())

        body = '\n'.join(call_strings)
        args = ',\n'.join(f'{buffer.get_type_string()} {buffer.name}' for buffer in self.buffers)

        _source = '\n'.join((
            '__kernel void compute_kernel(',
            indent(args, '\t'),
            ') {',
            indent(body, '\t'),
            '}'
        ))

        return _source

    def get_source(self) -> str:
        sources = {}
        for op in self.operations:
            _key = op.get_identifier()
            if _key in sources: continue
            sources[_key] = op.get_source()

        sources['_kernel'] = self.get_definition()

        return '\n\n'.join(sources.values())

    def compile(self, device: Device | None = None):
        _device = device or Device.get_active()

        _kernel = cl.Program(
            _device.get_cl_context(),
            self.get_source()
        ).build()

        self.cl_kernel = _kernel.compute_kernel
        self.cl_kernel.set_args(
            *(buffer.get_cl_buffer() for buffer in self.buffers)
        )