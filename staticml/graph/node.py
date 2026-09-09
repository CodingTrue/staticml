from staticml.common.lowering import LoweringContext
from staticml.graph.utils import _join_list, _indent_list
from staticml.graph.value import Value


class Node:
    def __init__(
            self,
            inputs: list[Value] | None = None,
            outputs: list[Value] | None = None
    ):
        self.inputs: list[Value] = inputs or []
        self.outputs: list[Value] = outputs or []

    def lower(self, context: LoweringContext):
        raise NotImplementedError

    def __repr__(self):
        args = [f'inputs = {_join_list(self.inputs)}', f'outputs = {_join_list(self.outputs)}']
        return f'{self.__class__.__name__}_{id(self):0x}(\n{_indent_list(args, '\n')}\n)'