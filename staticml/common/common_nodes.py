import numpy as np

from staticml.common import MatmulValueOperation, MapOperation
from staticml.common.common_ops import CommonValueOperation
from staticml.common.lowering import LoweringContext
from staticml.graph.node import Node
from staticml.graph.value import Value
from staticml.program import LaunchConfig
from staticml.shape import Shape


def _broadcast_shapes(left: Shape, right: Shape) -> Shape:
    for l_dim, r_dim in zip(left.as_tuple(), right.as_tuple()):
        if l_dim == r_dim or (l_dim == 1 or r_dim == 1): continue
        raise ValueError(f"Operands could not be broadcasted together with shapes {left} and {right}")

    return Shape(
        x=max(left.x, right.x),
        y=max(left.y, right.y),
        z=max(left.z, right.z)
    )

def _broadcast(a: Value, b: Value) -> tuple[Shape, Shape]:
    return (
        _broadcast_shapes(left=a.shape, right=b.shape),
        a.strides if a.size > b.size else b.strides
    )

class CommonABNode(Node):
    def __init__(self, a: Value, b: Value, symbol: str):
        self.a = a
        self.b = b
        self.symbol = symbol

        shape, strides = _broadcast(a=self.a, b=self.b)

        self.result = Value(
            shape=shape,
            strides=strides
        )

        super().__init__(
            inputs=[a, b],
            outputs=[self.result]
        )

    def lower(self, context: LoweringContext):
        context.add_operation(
            operation=CommonValueOperation(
                x_view=context.get_view(value=self.a),
                y_view=context.get_view(value=self.b),
                out_view = context.get_view(value=self.result),
                x_value=self.a,
                y_value=self.b,
                out_value=self.result,
                symbol=self.symbol
            ),
            launch_config=context.launch_config_like_shape_of(value=self.result)
        )

class MatmulNode(Node):
    def __init__(self, a: Value, b: Value):
        self.a = a
        self.b = b

        shape = Shape(
            x=b.shape.x,
            y=a.shape.y,
            z=max(a.shape.z, b.shape.z)
        )

        strides = Shape(
            x=1 if shape.x != -1 else -1,
            y=shape.x if shape.y != -1 else -1,
            z=shape.x * shape.y if shape.z != -1 else -1,
        )

        self.result = Value(
            shape=shape,
            strides=strides
        )

        super().__init__(
            inputs=[a, b],
            outputs=[self.result]
        )

    def lower(self, context: LoweringContext):
        context.add_operation(
            operation=MatmulValueOperation(
                x_view=context.get_view(value=self.a),
                y_view=context.get_view(value=self.b),
                out_view=context.get_view(value=self.result),
                x_value=self.a,
                y_value=self.b,
                out_value=self.result
            ),
            launch_config=context.launch_config_like_shape_of(value=self.result)
        )

class TransposeNode(Node):
    def __init__(self, x: Value, copy: bool = False):
        self.x = x

        if self.x.has_data:
            itemsize = self.x.data.dtype.itemsize
            data = self.x.data.copy() if copy else self.x.data.view()

            data = np.lib.stride_tricks.as_strided(
                x=data,
                shape=self.x.shape.as_tuple(),
                strides=(x * itemsize for x in self.x.strides.as_tuple())
            )

            result = Value(data=data)
        else:
            result = Value(
                shape=Shape(*self.x.shape.as_tuple(reverse=True)),
                strides=Shape(*self.x.strides.as_tuple(reverse=True))
            )

        self.result = result

        super().__init__(
            inputs=[self.x],
            outputs=[self.result]
        )

    def lower(self, context: LoweringContext):
        return

class MapNode(Node):
    def __init__(self, x: Value, expression: str):
        self.x = x
        self.expression = expression

        self.result = Value(
            shape=self.x.shape.copy(),
            strides=self.x.strides.copy()
        )

        super().__init__(
            inputs=[x],
            outputs=[self.result]
        )

    def lower(self, context: LoweringContext):
        context.add_operation(
            operation=MapOperation(
                x=context.get_view(value=self.x),
                out=context.get_view(value=self.result),
                expression=self.expression
            ),
            launch_config=LaunchConfig(x=self.x.size)
        )