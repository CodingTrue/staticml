import pyopencl as cl
from enum import Enum


def _get_device_name(device: cl.Device) -> str:
    name = ''

    if "Advanced Micro Devices, Inc." in device.vendor:
        name = device.board_name_amd
    else:
        name = device.name

    return name.strip()

class DeviceType(Enum):
    CPU = cl.device_type.CPU
    GPU = cl.device_type.GPU

class Device:
    _active: Device | None = None

    def __init__(self, device: cl.Device):
        self.device: cl.Device = device
        self._context: cl.Context = None
        self._queue: cl.CommandQueue = None

        self.profiling_enabled = False

    def __repr__(self):
        return f"Device('{self.name}', {self.device.global_mem_size / 1024**3:0.2f} GB)"

    def use(self) -> Device:
        self._context = cl.Context(devices=[self.device])
        self._queue = cl.CommandQueue(
            context=self._context,
            device=self.device,
            properties=cl.command_queue_properties.PROFILING_ENABLE * self.profiling_enabled
        )

        Device._active = self
        return self

    def with_profiling_enabled(self) -> Device:
        self.profiling_enabled = True
        return self

    @classmethod
    def active(cls) -> Device:
        if not cls._active:
            raise RuntimeError('No active device')
        return cls._active

    @property
    def name(self) -> str:
        return _get_device_name(device=self.device)

    @property
    def type(self) -> DeviceType:
        return DeviceType.CPU if self._device.type == DeviceType.CPU else DeviceType.GPU

    @property
    def context(self) -> cl.Context:
        if not self._context:
            raise RuntimeError('Context is not set')
        return self._context

    @property
    def queue(self) -> cl.CommandQueue:
        if not self._queue:
            raise RuntimeError('CommandQueue is not set')
        return self._queue