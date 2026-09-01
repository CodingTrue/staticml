from numbers import Number

from staticml.buffer import BufferView
from staticml.dtype import *
from staticml.operation import Operation, OperationBufferArg
from staticml.shape import Shape
from staticml.tensor import Tensor


def _normalize_strides(tensor: Tensor) -> Shape:
    shape_infl = tensor.shape.inflated

    return Shape(
        x=tensor.strides.x if shape_infl.x != 1 else 0,
        y=tensor.strides.y if shape_infl.y != 1 else 0,
        z=tensor.strides.z if shape_infl.z != 1 else 0
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
        out_shape = out_tensor.shape.inflated
        out_strides = _normalize_strides(out_tensor)
        x_strides = _normalize_strides(tensor=x_tensor)
        y_strides = _normalize_strides(tensor=y_tensor)

        super().__init__(name=f'common_binary', args=[
            OperationBufferArg(name='x', buffer=x.buffer, dtype=float32),
            OperationBufferArg(name='y', buffer=y.buffer, dtype=float32),
            OperationBufferArg(name='out', buffer=out.buffer, dtype=float32),
        ], body=[
            'int xid = get_global_id(0);',
            'int yid = get_global_id(1);',
            'int zid = get_global_id(2);',
            f'if (xid >= {out_shape.x} || yid >= {out_shape.y} || zid >= {out_shape.z}) return;',
            f'out[{out.offset} + zid * {out_strides.z} + yid * {out_strides.y} + xid * {out_strides.x}] = '
            f'x[{x.offset} + zid * {x_strides.z} + yid * {x_strides.y} + xid * {x_strides.x}] {symbol} '
            f'y[{y.offset} + zid * {y_strides.z} + yid * {y_strides.y} + xid * {y_strides.x}];'
        ])

class MatmulOperation(Operation):
    def __init__(
            self,
            x: BufferView,
            y: BufferView,
            x_tensor: Tensor,
            y_tensor: Tensor,
            out_tensor: Tensor,
            out: BufferView,
    ):
        # baked parameters will become a problem in the future once caching and proper operation-reuse is implemented
        out_shape = out_tensor.shape.inflated
        out_strides = _normalize_strides(out_tensor)
        x_strides = _normalize_strides(x_tensor)
        y_strides = _normalize_strides(y_tensor)

        super().__init__(name=f'matmul', args=[
            OperationBufferArg(name='x', buffer=x.buffer, dtype=float32),
            OperationBufferArg(name='y', buffer=y.buffer, dtype=float32),
            OperationBufferArg(name='out', buffer=out.buffer, dtype=float32),
        ], body=[
            'int xid = get_global_id(0);'
            'int yid = get_global_id(1);',
            'int zid = get_global_id(2);',
            f'if (xid >= {out_shape.x} || yid >= {out_shape.y} || zid >= {out_shape.z}) return;',
            'float result = 0.0;',
            f'for (int i = 0; i < {x_tensor.shape.x}; i++)' '{'
            f'  result = fma(',
            f'      x[{x.offset} + zid * {x_strides.z} + yid * {x_tensor.shape.x} + i],',
            f'      y[{y.offset} + zid * {y_strides.z} + i * {y_tensor.shape.x} + xid],',
            f'      result'
            f'  );',
            '}',
            f'out[{out.offset} + zid * {out_strides.z} + yid * {out_strides.y} + xid * {out_strides.x}] = result;'
        ])