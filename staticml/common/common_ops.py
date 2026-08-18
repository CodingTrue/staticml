from numbers import Number
from typing import Iterable

from staticml.buffer import Buffer, BufferView
from staticml.dtype import *
from staticml.operation import Operation, OperationArg, OperationBufferArg
from staticml.tensor import TensorShape


def _get_common_size(views: Iterable[BufferView]):
    return max(view.size for view in views)

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
        common_size = _get_common_size(views=(x,))

        super().__init__(name=f'axb', args=[
            OperationBufferArg(name='x', buffer=x.buffer, dtype=float32),
            OperationBufferArg(name='out', buffer=out.buffer, dtype=float32),
        ], body=[
            'int xid = get_global_id(0);',
            f'if (xid >= {common_size}) return;',
            f'out[{out.offset} + xid] = {a} {symbol} x[{x.offset} + xid] + {b};'
        ])

class AXBYOperation(Operation):
    def __init__(
            self,
            a: Number,
            x: BufferView,
            b: Number,
            y: BufferView,
            out: BufferView,
            x_strides: TensorShape,
            y_strides: TensorShape,
            out_shape: TensorShape,
            symbol: str,
    ):
        # baked parameters will become a problem in the future once caching and proper operation-reuse is implemented

        super().__init__(name=f'axby', args=[
            OperationBufferArg(name='x', buffer=x.buffer, dtype=float32),
            OperationBufferArg(name='y', buffer=y.buffer, dtype=float32),
            OperationBufferArg(name='out', buffer=out.buffer, dtype=float32),
        ], body=[
            'int xid = get_global_id(0);',
            'int yid = get_global_id(1);',
            f'if (xid >= {out_shape.x} || yid >= {out_shape.y}) return;',
            f'out[{out.offset} + yid * {out_shape.x} + xid] = ' 
            f'({a} * x[{x.offset} + yid * {x_strides.y} + xid * {x_strides.x}]) {symbol} '
            f'({b} * y[{y.offset} + yid * {y_strides.y} + xid * {y_strides.x}]);'
        ])

class MatmulOperation(Operation):
    def __init__(
            self,
            x: BufferView,
            y: BufferView,
            out_shape: TensorShape,
            out: BufferView,
    ):
        # baked parameters will become a problem in the future once caching and proper operation-reuse is implemented

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