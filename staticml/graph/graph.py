from textwrap import indent

from staticml.graph.lowering import LoweringContext
from staticml.graph.value import Value


def _join_list(a, delimiter = ', '):
    return delimiter.join(str(x) for x in a)

def _indent_list(a, delimiter = ', '):
    return indent(_join_list(a, delimiter), '\t')

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

class Graph:
    def __init__(self):
        self.nodes: list[Node] = []

    def __repr__(self):
        return f'Graph_{id(self):0x}(\n{_indent_list(self.nodes, ',\n')}\n)'

    def add_node(self, node: Node) -> Graph:
        if node not in self.nodes:
            self.nodes.append(node)
        return self