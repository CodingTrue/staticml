from numbers import Number

from staticml.operation import *
from staticml.tensor import Tensor, TensorOperation

def _align_scalar(a, b) -> tuple:
    if isinstance(a, Number):
        return b, a
    return a, b

def _handle_add(tensor: Tensor) -> Operation:
    a, b = _align_scalar(*tensor.children)
    if isinstance(b, Tensor):
        return AXPBYZOperation(1, a, 1, b, tensor)

    return AXBZOperation(1, a, b, tensor)

def _handle_sub(tensor: Tensor) -> Operation:
    a, b = _align_scalar(*tensor.children)
    if isinstance(b, Tensor):
        return AXPBYZOperation(1, a, -1, b, tensor)

    return AXBZOperation(1, a, -b, tensor)

def _handle_mul(tensor: Tensor) -> Operation:
    a, b = _align_scalar(*tensor.children)
    if isinstance(b, Tensor):
        return XMULYOperation(a, b, tensor)

    return AXBZOperation(b, a, 0, tensor)

def _handle_div(tensor: Tensor) -> Operation:
    a, b = _align_scalar(*tensor.children)
    if isinstance(b, Tensor):
        return XDIVYOperation(a, b, tensor)

    return AXBZOperation(1 / b, a, 0, tensor)

def _handle_matmul(tensor: Tensor) -> Operation:
    a, b = tensor.children
    return MATMULOperation(a, b, tensor)

TENSOROP_HANDLES = {
    TensorOperation.ADD:        _handle_add,
    TensorOperation.SUBTRACT:   _handle_sub,
    TensorOperation.MULTIPLY:   _handle_mul,
    TensorOperation.DIVIDE:     _handle_div,
    TensorOperation.MATMUL:     _handle_matmul,
}

def lower_tensor(tensor: Tensor) -> Operation:
    return TENSOROP_HANDLES[tensor.operation](tensor)