from dataclasses import dataclass
from numbers import Number
from typing import Callable

from staticml.common import CommonBinaryOperation, AXBOperation, Allocator, MatmulOperation, MapOperation
from staticml.buffer import BufferView
from staticml.operation import Operation
from staticml.program import LaunchConfig
from staticml.tensor import Tensor, TensorOperation


type TensorArg = Tensor | Number

@dataclass
class LoweringContext:
    tensor: Tensor
    a: TensorArg
    b: TensorArg
    a_and_b_tensors: bool
    aligned_a: Tensor
    aligned_b: TensorArg
    view_callback: Callable[[Tensor], BufferView]
    allocator: Allocator
    common_size: int

    out_view: BufferView = None
    out_operation: Operation = None
    out_launch_config: LaunchConfig = None

    def get_view(self, tensor: Tensor) -> BufferView:
        return self.view_callback(tensor)

    @property
    def simple_view(self) -> BufferView:
        v = self.allocator.allocate(size=self.common_size)

        self.out_view = v
        return v

def _get_common_size(views: Iterable[BufferView]):
    return max(view.size for view in views)

def _align_scalar(a: TensorArg, b: TensorArg) -> tuple[Tensor, TensorArg]:
    if isinstance(a, Number):
        return b, a
    return a, b

def _common_op(context: LoweringContext, symbol: str):
    context.out_operation = CommonBinaryOperation(
        x=context.get_view(tensor=context.a),
        y=context.get_view(tensor=context.b),
        out=context.simple_view,
        x_tensor=context.a,
        y_tensor=context.b,
        out_tensor=context.tensor,
        symbol=symbol
    )

    context.out_launch_config = LaunchConfig(*context.tensor.shape.as_tuple())

def _handle_add(context: LoweringContext):
    if context.a_and_b_tensors:
        _common_op(context=context, symbol='+')
    else:
        context.out_operation = AXBOperation(
            a=1, b=context.aligned_b,
            x=context.get_view(tensor=context.aligned_a),
            out=context.simple_view,
            symbol='*'
        )

def _handle_sub(context: LoweringContext):
    if context.a_and_b_tensors:
        _common_op(context=context, symbol='-')
    else:
        if isinstance(context.a, Number):
            a_sign, b_sign = -1, 1
        else:
            a_sign, b_sign = 1, -1

        context.out_operation = AXBOperation(
            a=a_sign, b=context.aligned_b * b_sign,
            x=context.get_view(tensor=context.aligned_a),
            out=context.simple_view,
            symbol='*'
        )

def _handle_mul(context: LoweringContext):
    if context.a_and_b_tensors:
        _common_op(context=context, symbol='*')
    else:
        context.out_operation = AXBOperation(
            a=context.aligned_b, b=0,
            x=context.get_view(tensor=context.aligned_a),
            out=context.simple_view,
            symbol='*'
        )

def _handle_div(context: LoweringContext):
    if context.a_and_b_tensors:
        _common_op(context=context, symbol='/')
    else:
        symbol = '*'
        if isinstance(context.b, Number):
            scalar = 1 / context.b
        else:
            scalar = context.a
            symbol = '/'

        context.out_operation = AXBOperation(
            a=scalar, b=0,
            x=context.get_view(tensor=context.aligned_a),
            out=context.simple_view,
            symbol=symbol
        )

def _handle_matmul(context: LoweringContext):
    shape = context.tensor.shape.inflated
    out_view = context.allocator.allocate(size=shape.x * shape.y * shape.z)

    context.out_operation = MatmulOperation(
        x=context.get_view(tensor=context.a),
        y=context.get_view(tensor=context.b),
        x_tensor=context.a,
        y_tensor=context.b,
        out_tensor=context.tensor,
        out=out_view
    )

    context.out_view = out_view
    context.out_launch_config = LaunchConfig(x=shape.x, y=shape.y, z=shape.z)

def _handle_map(context: LoweringContext):
    context.out_operation = MapOperation(
        x=context.get_view(tensor=context.a),
        expression=context.b,
        out=context.simple_view
    )

HANDLES = {
    TensorOperation.ADD: _handle_add,
    TensorOperation.SUB: _handle_sub,
    TensorOperation.MUL: _handle_mul,
    TensorOperation.DIV: _handle_div,
    TensorOperation.MATMUL: _handle_matmul,
    TensorOperation.MAP: _handle_map
}

def lower_tensor(
        tensor: Tensor,
        view_callback: Callable[[Tensor], BufferView],
        allocator: Allocator
) -> list[LoweringResult]:
    to, *args = tensor.args

    if to in (
            TensorOperation.ADD, TensorOperation.SUB, TensorOperation.MUL, TensorOperation.DIV,
            TensorOperation.MATMUL
    ):
        a, b, aligned_a, aligned_b = *args, *_align_scalar(*args)
    else:
        a, b, aligned_a, aligned_b = args[0], args[1], None, None

    lc = LoweringContext(
        tensor=tensor,
        a=a, b=b,
        a_and_b_tensors=all(Tensor.is_tensor(o=arg) for arg in (a, b)),
        aligned_a=aligned_a, aligned_b=aligned_b,
        view_callback=view_callback,
        allocator=allocator,
        common_size=_get_common_size(views=(view_callback(arg) for arg in (a, b) if Tensor.is_tensor(o=arg)))
    )

    HANDLES[to](lc)

    if not lc.out_launch_config:
        lc.out_launch_config = LaunchConfig(x=lc.common_size)

    return lc
