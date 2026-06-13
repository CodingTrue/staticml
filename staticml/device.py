from enum import Enum

import pyopencl as cl

def _get_device_name(cl_device: cl.Device) -> str:
    _name = cl_device.board_name_amd if 'Advanced Micro Devices, Inc.' in cl_device.vendor else cl_device.name
    return str(_name).strip()

class DeviceType(Enum):
    ALL = cl.device_type.ALL
    GPU = cl.device_type.GPU
    CPU = cl.device_type.CPU

class Device:
    _active: Device = None

    def __init__(self, cl_device: cl.Device):
        self.cl_device: cl.Device = cl_device
        self.cl_context: cl.Context = None
        self.cl_queue: cl.CommandQueue = None

        self._display_name = _get_device_name(cl_device=self.cl_device)

    def get_cl_context(self) -> cl.Context:
        if self.cl_context is None:
            raise RuntimeError('Device context was not initialized yet')
        return self.cl_context

    def get_cl_queue(self) -> cl.Context:
        if self.cl_context is None:
            raise RuntimeError('Device queue was not initialized yet')
        return self.cl_queue

    def get_name(self) -> str:
        return self._display_name

    def use(self) -> Device:
        Device._active = self

        self.cl_context: cl.Context = cl.Context([self.cl_device])
        self.cl_queue: cl.CommandQueue = cl.CommandQueue(self.cl_context, device=self.cl_device)

        return self

    @classmethod
    def get_active(cls) -> Device:
        if not cls._active:
            raise RuntimeError('No device is currently set as active')
        return cls._active

    @staticmethod
    def get_device(hint: str | None = None, device_type: DeviceType = DeviceType.ALL) -> Device:
        cl_device: cl.Device = None
        hint = hint or ''
        _break = False

        for platform in cl.get_platforms():
            if _break: break
            for device in platform.get_devices(device_type=device_type.value):
                if hint in _get_device_name(cl_device=device):
                    cl_device = device
                    _break = True

                if _break: break
                cl_device = device
        return Device(cl_device=cl_device)