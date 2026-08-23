from numbers import Number

from staticml.buffer import BufferView
from staticml.dtype import *
from staticml.operation import Operation, OperationBufferArg
from staticml.tensor import Tensor, TensorShape


def _normalize_strides(tensor: Tensor) -> TensorShape:
    return TensorShape(
        x=0 if tensor.shape.x == 1 else 1,
        y=0 if tensor.shape.y == 1 else tensor.shape.x,
        z=tensor.shape.x * tensor.shape.y
    )

class AXBOperation(Operation):
    def __init__(
            self,
            a: Number,
            x: BufferView,
            b: Number,
            out: BufferView,
            symbol: str,
    ):
        # baked parameters will become a problem in the future once caching and proper operation-reuse is implemented

        super().__init__(name=f'axb', args=[
            OperationBufferArg(name='x', buffer=x.buffer, dtype=float32),
            OperationBufferArg(name='out', buffer=out.buffer, dtype=float32),
        ], body=[
            'int xid = get_global_id(0);',
            f'if (xid >= {x.size}) return;',
            f'out[{out.offset} + xid] = {a} {symbol} x[{x.offset} + xid] + {b};'
        ])

class CommonBinaryOperation(Operation):
    def __init__(
            self,
            x: BufferView,
            y: BufferView,
            out: BufferView,
            x_tensor: Tensor,
            y_tensor: Tensor,
            out_tensor: Tensor,
            symbol: str,
    ):
        # baked parameters will become a problem in the future once caching and proper operation-reuse is implemented
        out_shape = out_tensor.shape
        x_strides = _normalize_strides(tensor=x_tensor)
        y_strides = _normalize_strides(tensor=y_tensor)

        super().__init__(name=f'common_binary', args=[
            OperationBufferArg(name='x', buffer=x.buffer, dtype=float32),
            OperationBufferArg(name='y', buffer=y.buffer, dtype=float32),
            OperationBufferArg(name='out', buffer=out.buffer, dtype=float32),
        ], body=[
            'int xid = get_global_id(0);',
            'int yid = get_global_id(1);',
            f'if (xid >= {out_shape.x} || yid >= {out_shape.y}) return;',
            f'out[{out.offset} + yid * {out_shape.x} + xid] = ' 
            f'x[{x.offset} + yid * {x_strides.y} + xid * {x_strides.x}] {symbol} '
            f'y[{y.offset} + yid * {y_strides.y} + xid * {y_strides.x}];'
        ])

class MatmulOperation(Operation):
    def __init__(
            self,
            x: BufferView,
            y: BufferView,
            out_tensor: Tensor,
            out: BufferView,
    ):
        # baked parameters will become a problem in the future once caching and proper operation-reuse is implemented
        out_shape = out_tensor.shape

        super().__init__(name=f'matmul', args=[
            OperationBufferArg(name='x', buffer=x.buffer, dtype=float32),
            OperationBufferArg(name='y', buffer=y.buffer, dtype=float32),
            OperationBufferArg(name='out', buffer=out.buffer, dtype=float32),
        ], body=[
            'int xid = get_global_id(0);'
            'int yid = get_global_id(1);',
            f'if (xid >= {out_shape.x} || yid >= {out_shape.y}) return;',
            'float result = 0.0;',
            f'for (int i = 0; i < {out_shape.y}; i++)' ' {',
            f'  result += x[{x.offset} + i + yid * {out_shape.y}] * y[{y.offset} + xid + i * {out_shape.x}];',
            '}',
            f'out[{out.offset} + yid * {out_shape.x} + xid] = result;'
        ])