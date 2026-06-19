import re

from hashlib import blake2s
from numbers import Number
from textwrap import indent
from typing import Any

from staticml.buffer import Allocator, BufferRange
from staticml.tensor import Tensor


def _hash(s: str) -> str:
    return blake2s(s.encode(), digest_size=6).hexdigest()

class Operation:
    def __init__(self, identifier: str = '', arguments: dict[str, Any] | None = None, body: str = ''):
        self.arguments = arguments or {}
        self.body = body
        self.identifier = identifier or self.__class__.__name__.lower()

    def get_identifier(self) -> str:
        extension = ''

        for value in self.arguments.values():
            if isinstance(value, Tensor):
                extension += value.buffer_view.buffer.asq.name.lower()
                continue
            extension += type(value).__name__[0].lower()

        extension = _hash(extension)
        return f'op_{self.identifier}_{extension}'

    def get_source(self) -> str:
        arg_defs = []
        tensor_fields = []

        for field, value in self.arguments.items():
            if isinstance(value, Tensor):
                tensor_fields.append(field)

                _buffer = value.buffer_view.buffer

                arg_defs.append(f'{_buffer.get_type_string()} {field}_data')
                arg_defs.append(f'int {field}_size')
                arg_defs.append(f'int {field}_offset')
                continue

            arg_defs.append(f'{type(value).__name__} {field}')

        arg_header = ',\n'.join(arg_defs)
        body = self.body

        for tensor_field in tensor_fields:
            for reg, repl in [
                (rf'\b{tensor_field}\.', f'{tensor_field}_'),
                (rf'{tensor_field}\[', f'{tensor_field}_data[')
            ]:
                body = re.sub(reg, repl, body)

        source = '\n'.join((
            f'void {self.get_identifier()} (',
            indent(arg_header, '\t'),
            ') {',
            indent(body, '\t'),
            '}'
        ))

        return source

    def get_call_string(self) -> str:
        parsed_args = []

        for value in self.arguments.values():
            if isinstance(value, Number):
                parsed_args.append(str(value).lower())
            elif isinstance(value, Tensor):
                if not value.is_allocated():
                    raise RuntimeError("Can't generate a call string with unallocated tensors")

                _buffer: BufferRange = value.buffer_view

                parsed_args.append(_buffer.buffer.name)
                parsed_args.append(str(_buffer.size))
                parsed_args.append(str(_buffer.offset))
            else:
                raise ValueError(f"Could not find translation for '{_type}'")

        argstring = ', '.join(parsed_args)
        call_string = f'{self.get_identifier()}({argstring});'

        return call_string

    def allocate(self, allocator: Allocator):
        return

    def get_work_size(self) -> tuple[int, ...]:
        return (-1, -1, -1)

class AXBZOperation(Operation):
    def __init__(self, a: Number, x: Tensor, b: Number, z: Tensor):
        super().__init__(identifier='axb', arguments={
            'a': float(a),
            'x': x,
            'b': float(b),
            'z': z
        }, body='int xid = get_global_id(0);\nz[z.offset + xid] = a * x[x.offset + xid] + b;')

        self.x = x
        self.z = z

    def allocate(self, allocator: Allocator):
        _buffer = allocator.allocate(size=self.x.get_size())

        self.z.set_buffer_view(view=_buffer)

    def get_work_size(self) -> tuple[int, ...]:
        return (self.x.get_size(), 1, 1)

class AXPBYZOperation(Operation):
    def __init__(self, a: Number, x: Tensor, b: Number, y: Tensor, z: Tensor):

        super().__init__(identifier='axpby', arguments={
            'a': float(a),
            'x': x,
            'b': float(b),
            'y': y,
            'z': z
        }, body='int xid = get_global_id(0);\nz[z.offset + xid] = a * x[x.offset + xid] + b * y[y.offset + xid];')
        self.min_size = min(x.get_size(), y.get_size())
        self.z = z

    def allocate(self, allocator: Allocator):
        _buffer = allocator.allocate(size=self.min_size)

        self.z.set_buffer_view(view=_buffer)

    def get_work_size(self) -> tuple[int, ...]:
        return (self.min_size, 1, 1)

class XMULYOperation(Operation):
    def __init__(self, x: Tensor, y: Tensor, z: Tensor):

        super().__init__(identifier='xmuly', arguments={
            'x': x,
            'y': y,
            'z': z
        }, body='int xid = get_global_id(0);\nz[z.offset + xid] = x[x.offset + xid] * y[y.offset + xid];')
        self.min_size = min(x.get_size(), y.get_size())
        self.z = z

    def allocate(self, allocator: Allocator):
        _buffer = allocator.allocate(size=self.min_size)

        self.z.set_buffer_view(view=_buffer)

    def get_work_size(self) -> tuple[int, ...]:
        return (self.min_size, 1, 1)

class XDIVYOperation(Operation):
    def __init__(self, x: Tensor, y: Tensor, z: Tensor):

        super().__init__(identifier='xdivy', arguments={
            'x': x,
            'y': y,
            'z': z
        }, body='int xid = get_global_id(0);\nz[z.offset + xid] = x[x.offset + xid] / y[y.offset + xid];')
        self.min_size = min(x.get_size(), y.get_size())
        self.z = z

    def allocate(self, allocator: Allocator):
        _buffer = allocator.allocate(size=self.min_size)

        self.z.set_buffer_view(view=_buffer)

    def get_work_size(self) -> tuple[int, ...]:
        return (self.min_size, 1, 1)