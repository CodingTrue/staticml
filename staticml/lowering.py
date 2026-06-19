from numbers import Number

from staticml.operation import Operation, AXBZOperation
from staticml.tensor import Tensor, TensorOperation

def _align_scalar(a, b) -> tuple:
    if isinstance(a, Number):
        return b, a
    return a, b

def _handle_add(tensor: Tensor) -> Operation:
    a, b = _align_scalar(*tensor.children)
    if isinstance(b, Tensor):
        raise NotImplementedError("Tensor + tensor operations are not supported yet")

    return AXBZOperation(1, a, b, tensor)

def _handle_sub(tensor: Tensor) -> Operation:
    a, b = _align_scalar(*tensor.children)
    if isinstance(b, Tensor):
        raise NotImplementedError("Tensor - tensor operations are not supported yet")

    return AXBZOperation(1, a, -b, tensor)

def _handle_mul(tensor: Tensor) -> Operation:
    a, b = _align_scalar(*tensor.children)
    if isinstance(b, Tensor):
        raise NotImplementedError("Tensor * tensor operations are not supported yet")

    return AXBZOperation(b, a, 0, tensor)

def _handle_div(tensor: Tensor) -> Operation:
    a, b = _align_scalar(*tensor.children)
    if isinstance(b, Tensor):
        raise NotImplementedError("Tensor / tensor operations are not supported yet")

    return AXBZOperation(1 / b, a, 0, tensor)

TENSOROP_HANDLES = {
    TensorOperation.ADD:        _handle_add,
    TensorOperation.SUBTRACT:   _handle_sub,
    TensorOperation.MULTIPLY:   _handle_mul,
    TensorOperation.DIVIDE:     _handle_div,
}

def lower_tensor(tensor: Tensor) -> Operation:
    return TENSOROP_HANDLES[tensor.operation](tensor)