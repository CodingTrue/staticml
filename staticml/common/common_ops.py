from numbers import Number
from typing import Iterable

from staticml.buffer import Buffer, BufferView
from staticml.dtype import *
from staticml.operation import Operation, OperationArg, OperationBufferArg


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
            symbol: str,
    ):
        # baked parameters will become a problem in the future once caching and proper operation-reuse is implemented
        common_size = _get_common_size(views=(x, y))

        super().__init__(name=f'axby', args=[
            OperationBufferArg(name='x', buffer=x.buffer, dtype=float32),
            OperationBufferArg(name='y', buffer=y.buffer, dtype=float32),
            OperationBufferArg(name='out', buffer=out.buffer, dtype=float32),
        ], body=[
            'int xid = get_global_id(0);',
            f'if (xid >= {common_size}) return;',
            f'out[{out.offset} + xid] = ({a} * x[{x.offset} + xid]) {symbol} ({b} * y[{y.offset} + xid]);'
        ])
