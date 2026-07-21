import pyopencl as cl
from enum import Enum


def _get_device_name(device: cl.Device) -> str:
    if "Advanced Micro Devices, Inc." in device.vendor:
        return device.board_name_amd
    return device.name

class DeviceType(Enum):
    CPU = cl.device_type.CPU
    GPU = cl.device_type.GPU

class Device:
    def __init__(self, device: cl.Device):
        self._device: cl.Device = device

        self._name = _get_device_name(device=self._device)
        self._type: DeviceType = DeviceType.CPU if self._device.type == DeviceType.CPU else DeviceType.GPU
        self._total_bytes = self._device.global_mem_size

    def __repr__(self):
        return f"Device('{self._name}', {self._total_bytes / 1024**3:0.2f} GB)"

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> DeviceType:
        return self._type