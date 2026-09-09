import re

from staticml.graph.value import Value
from staticml.buffer import BufferView
from staticml.dtype import *
from staticml.operation import Operation, OperationBufferArg
from staticml.shape import Shape


def _normalize_strides(value: Value) -> Shape:
    shape_infl = value.shape.inflated

    return Shape(
        x=value.strides.x if shape_infl.x != 1 else 0,
        y=value.strides.y if shape_infl.y != 1 else 0,
        z=value.strides.z if shape_infl.z != 1 else 0
    )

class MapOperation(Operation):
    def __init__(
            self,
            x: BufferView,
            expression: str,
            out: BufferView,
    ):
        # baked parameters will become a problem in the future once caching and proper operation-reuse is implemented
        expression = re.sub(r'(?<!\w)x(?!\w)', f'x[{x.offset} + xid]', expression)

        super().__init__(name=f'map', args=[
            OperationBufferArg(name='x', buffer=x.buffer, dtype=float32),
            OperationBufferArg(name='out', buffer=out.buffer, dtype=float32),
        ], body=[
            'int xid = get_global_id(0);',
            f'if (xid >= {x.size}) return;',
            f'out[{out.offset} + xid] = {expression};'
        ])

class CommonValueOperation(Operation):
    def __init__(
            self,
            x_view: BufferView,
            y_view: BufferView,
            out_view: BufferView,
            x_value: Value,
            y_value: Value,
            out_value: Value,
            symbol: str
    ):
        # baked parameters will become a problem in the future once caching and proper operation-reuse is implemented
        out_shape = out_value.shape.inflated
        out_strides = _normalize_strides(value=out_value)
        x_strides = _normalize_strides(value=x_value)
        y_strides = _normalize_strides(value=y_value)

        super().__init__(name=f'common_binary', args=[
            OperationBufferArg(name='x', buffer=x_view.buffer, dtype=float32),
            OperationBufferArg(name='y', buffer=y_view.buffer, dtype=float32),
            OperationBufferArg(name='out', buffer=out_view.buffer, dtype=float32),
        ], body=[
            'int xid = get_global_id(0);',
            'int yid = get_global_id(1);',
            'int zid = get_global_id(2);',
            f'if (xid >= {out_shape.x} || yid >= {out_shape.y} || zid >= {out_shape.z}) return;',
            f'out[{out_view.offset} + zid * {out_strides.z} + yid * {out_strides.y} + xid * {out_strides.x}] = '
            f'x[{x_view.offset} + zid * {x_strides.z} + yid * {x_strides.y} + xid * {x_strides.x}] {symbol} '
            f'y[{y_view.offset} + zid * {y_strides.z} + yid * {y_strides.y} + xid * {y_strides.x}];'
        ])

class MatmulValueOperation(Operation):
    def __init__(
            self,
            x_view: BufferView,
            y_view: BufferView,
            out_view: BufferView,
            x_value: Value,
            y_value: Value,
            out_value: Value
    ):
        # baked parameters will become a problem in the future once caching and proper operation-reuse is implemented
        out_shape = out_value.shape.inflated
        out_strides = _normalize_strides(value=out_value)
        x_strides = _normalize_strides(value=x_value)
        y_strides = _normalize_strides(value=y_value)

        super().__init__(name=f'matmul', args=[
            OperationBufferArg(name='x', buffer=x_view.buffer, dtype=float32),
            OperationBufferArg(name='y', buffer=y_view.buffer, dtype=float32),
            OperationBufferArg(name='out', buffer=out_view.buffer, dtype=float32),
        ], body=[
            'int xid = get_global_id(0);',
            'int yid = get_global_id(1);',
            'int zid = get_global_id(2);',
            f'if (xid >= {out_shape.x} || yid >= {out_shape.y} || zid >= {out_shape.z}) return;',
            'float result = 0.0;',
            f'for (int i = 0; i < {x_value.shape.x}; i++)' '{'
            f'  result = fma(',
            f'      x[{x_view.offset} + zid * {x_strides.z} + yid * {x_strides.y} + i * {x_strides.x}],',
            f'      y[{y_view.offset} + zid * {y_strides.z} + i * {y_strides.y} + xid * {y_strides.x}],',
            f'      result',
            f'  );',
            '}',
            f'out[{out_view.offset} + zid * {out_strides.z} + yid * {out_strides.y} + xid * {out_strides.x}] = result;'
        ])