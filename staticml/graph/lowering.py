from dataclasses import dataclass
from typing import Callable

from staticml.graph.value import Value
from staticml.buffer import BufferView
from staticml.operation import Operation
from staticml.program import LaunchConfig
from staticml.shape import Shape


@dataclass
class OperationEntry:
    operation: Operation
    launch_config: LaunchConfig

@dataclass
class LoweringContext:
    view_callback: Callable[[Value], BufferView]
    operations: list[OperationEntry]

    def get_view(self, value: Value) -> BufferView | None:
        return self.view_callback(value)

    def add_operation(self, operation: Operation, launch_config: LaunchConfig) -> LoweringContext:
        if operation not in self.operations:
            self.operations.append(OperationEntry(
                operation=operation,
                launch_config=launch_config
            ))
        return self

    def shape_as_launch_config(self, shape: Shape) -> LaunchConfig:
        return LaunchConfig(*shape.as_tuple())

    def launch_config_like_shape_of(self, value: Value) -> LaunchConfig:
        return self.shape_as_launch_config(shape=value.shape)