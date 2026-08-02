import pyopencl as cl

from textwrap import indent

from staticml.buffer import Buffer, BASE_DTYPE
from staticml.device import Device
from staticml.operation import Operation, OperationBufferArg


class LaunchConfig:
    def __init__(self, x: int = 1, y: int = 1, z: int = 1):
        self.x = max(x, 1)
        self.y = max(y, 1)
        self.z = max(z, 1)

    def as_tuple(self) -> tuple[int, int, int]:
        return self.x, self.y, self.z

class Kernel:
    def __init__(self, operations: list[Operation] | None = None, launch_config: LaunchConfig | None = None):
        self.operations = operations or []
        self.launch_config = launch_config

        self._kernel: cl.Kernel = None
        self._device: Device = None

    def _check_compiled_status(self):
        if not self.is_compiled:
            raise RuntimeError('Kernel is not compiled')

    def add_operation(self, operation: Operation):
        self.operations.append(operation)

    def compile(self, device: Device | None = None) -> Kernel:
        if self.is_compiled:
            raise RuntimeError('Kernel is already compiled')

        device = device or Device.active()

        source = []
        buffers: list[Buffer] = []
        call_strings = []

        for operation in self.operations:
            source.append(operation.source_string)
            call_strings.append(operation.call_string)

            for arg in operation.args:
                if not isinstance(arg, OperationBufferArg) or arg.value in buffers: continue
                buffers.append(arg.value)

        buffer_strings = ',\n'.join(f'{buffer.asq.value} {BASE_DTYPE.c_name}* {buffer.name}' for buffer in buffers)
        call_strings = '\n'.join(call_strings)

        source.append('\n'.join((
            '__kernel void compute_kernel(',
            indent(buffer_strings, '\t'),
            ') {',
            indent(call_strings, '\t'),
            '}'
        )))

        source = '\n'.join(source)

        program = cl.Program(
            arg1=device.context,
            arg2=source
        ).build()

        self._kernel = program.compute_kernel
        self._kernel.set_args(*[buffer.buffer for buffer in buffers])

        self._device = device
        return self

    def run(self):
        self._check_compiled_status()
        if self.launch_config is None:
            raise RuntimeError('Launch config is not set')

        cl.enqueue_nd_range_kernel(
            queue=self._device.queue,
            kernel=self._kernel,
            global_work_size=self.launch_config.as_tuple(),
            local_work_size=None
        )

    @property
    def is_compiled(self) -> bool:
        return self._kernel is not None

    @property
    def kernel(self) -> cl.Kernel:
        self._check_compiled_status()
        return self._kernel

class Program:
    def __init__(self, kernels: list[Kernel] | None = None):
        self._kernels = kernels or []
        self._device = None

    def build(self, device: Device | None = None) -> Program:
        device = device or Device.active()

        for kernel in self._kernels:
            kernel.compile(device=device)

        self._device = device
        return self

    def run(self) -> Program:
        for kernel in self._kernels:
            kernel.run()

        self._device.queue.finish()
        return self