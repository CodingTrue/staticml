from staticml.graph.node import Node
from staticml.graph.utils import _indent_list


class Graph:
    def __init__(self):
        self.nodes: list[Node] = []

    def __repr__(self):
        return f'Graph_{id(self):0x}(\n{_indent_list(self.nodes, ',\n')}\n)'

    def add_node(self, node: Node) -> Graph:
        if node not in self.nodes:
            self.nodes.append(node)
        return self