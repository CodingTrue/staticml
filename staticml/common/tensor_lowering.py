from dataclasses import dataclass
from numbers import Number
from typing import Callable

from staticml.common import AXBYOperation, AXBOperation, Allocator, MatmulOperation
from staticml.buffer import BufferView
from staticml.operation import Operation
from staticml.program import LaunchConfig
from staticml.tensor import Tensor, TensorOperation, TensorShape

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

def _handle_add(context: LoweringContext):
    if context.a_and_b_tensors:
        out_shape = max(context.a.shape.as_tuple(), context.b.shape.as_tuple())

        context.out_operation = AXBYOperation(
            a=1, b=1,
            x=context.get_view(tensor=context.a),
            y=context.get_view(tensor=context.b),
            out=context.simple_view,
            x_strides=context.a.strides,
            y_strides=context.b._strides,
            out_shape=TensorShape(*out_shape),
            symbol='+'
        )

        context.out_launch_config = LaunchConfig(*out_shape)
    else:
        context.out_operation = AXBOperation(
            a=1, b=context.aligned_b,
            x=context.get_view(tensor=context.aligned_a),
            out=context.simple_view,
            symbol='*'
        )

def _handle_sub(context: LoweringContext):
    if context.a_and_b_tensors:
        out_shape = max(context.a.shape.as_tuple(), context.b.shape.as_tuple())

        context.out_operation = AXBYOperation(
            a=1, b=-1,
            x=context.get_view(tensor=context.a),
            y=context.get_view(tensor=context.b),
            out=context.simple_view,
            x_strides=context.a.strides,
            y_strides=context.b._strides,
            out_shape=TensorShape(*out_shape),
            symbol='+'
        )

        context.out_launch_config = LaunchConfig(*out_shape)
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
        out_shape = max(context.a.shape.as_tuple(), context.b.shape.as_tuple())

        context.out_operation = AXBYOperation(
            a=1, b=1,
            x=context.get_view(tensor=context.a),
            y=context.get_view(tensor=context.b),
            out=context.simple_view,
            x_strides=context.a.strides,
            y_strides=context.b._strides,
            out_shape=TensorShape(*out_shape),
            symbol='*'
        )

        context.out_launch_config = LaunchConfig(*out_shape)
    else:
        context.out_operation = AXBOperation(
            a=context.aligned_b, b=0,
            x=context.get_view(tensor=context.aligned_a),
            out=context.simple_view,
            symbol='*'
        )

def _handle_div(context: LoweringContext):
    if context.a_and_b_tensors:
        out_shape = max(context.a.shape.as_tuple(), context.b.shape.as_tuple())

        context.out_operation = AXBYOperation(
            a=1, b=1,
            x=context.get_view(tensor=context.a),
            y=context.get_view(tensor=context.b),
            out=context.simple_view,
            x_strides=context.a.strides,
            y_strides=context.b._strides,
            out_shape=TensorShape(*out_shape),
            symbol='/'
        )

        context.out_launch_config = LaunchConfig(*out_shape)
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
    shape = context.tensor.shape
    out_view = context.allocator.allocate(size=shape.x * shape.y)

    context.out_operation = MatmulOperation(
        x=context.get_view(tensor=context.a),
        y=context.get_view(tensor=context.b),
        out_shape=shape,
        out=out_view
    )

    context.out_view = out_view
    context.out_launch_config = LaunchConfig(x=shape.x, y=shape.y)

HANDLES = {
    TensorOperation.ADD: _handle_add,
    TensorOperation.SUB: _handle_sub,
    TensorOperation.MUL: _handle_mul,
    TensorOperation.DIV: _handle_div,
    TensorOperation.MATMUL: _handle_matmul,
}

def lower_tensor(
        tensor: Tensor,
        view_callback: Callable[[Tensor], BufferView],
        allocator: Allocator
) -> LoweringContext:
    to, *args = tensor.args

    if len(args) != 2:
        raise RuntimeError('Tensor args must contain exactly two operands')

    a, b, aligned_a, aligned_b = *args, *_align_scalar(*args)

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
