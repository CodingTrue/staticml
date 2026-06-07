from textwrap import indent

from staticml.buffer import Buffer
from staticml.operation import Operation

class Kernel:
    def __init__(self, buffers: list[Buffer] | None = None, operations: list[Operation] | None = None):
        self.buffers: list[Buffer] = buffers or []
        self.operations: list[Operation] = operations or []

    def add_operation(self, operation: Operation):
        self.operations.append(operation)

    def get_definition(self) -> str:
        call_strings = []
        for op in self.operations:
            call_strings.append(op.get_call_string())

        body = '\n'.join(call_strings)
        args = ',\n'.join(f'{buffer.get_type_string()} {buffer.name}' for buffer in self.buffers)

        _source = '\n'.join((
            '__kernel void compute_kernel(',
            indent(args, '\t'),
            ') {',
            indent(body, '\t'),
            '}'
        ))

        return _source

    def get_source(self) -> str:
        sources = {}
        for op in self.operations:
            _key = op.get_identifier()
            if _key in sources: continue
            sources[_key] = op.get_source()

        sources['_kernel'] = self.get_definition()

        return '\n\n'.join(sources.values())