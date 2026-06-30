import re

from hashlib import blake2s
from numbers import Number
from textwrap import indent
from typing import Any

from staticml.buffer import Allocator, BufferRange
from staticml.tensor import Tensor


def _operation_layout(a: Tensor, b: Tensor) -> tuple[int, int, int]:
    a_shape = a.get_shape()
    b_shape = b.get_shape()

    is_greater = a_shape[1] > b_shape[1]

    size = max(a.get_size(), b.get_size())
    a_row, b_row = (size if is_greater else a_shape[2], b_shape[2] if is_greater else size)

    return size, a_row, b_row

def _hash(s: str) -> str:
    return blake2s(s.encode(), digest_size=6).hexdigest()

def _shape_size(shape: tuple[int, ...]) -> int:
    result = None
    for dim in shape:
        if result is None:
            result = dim
        else:
            result *= dim
    return result

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
        self.size, x_row, y_row = _operation_layout(x, y)

        super().__init__(identifier='axpby', arguments={
            'a': float(a),
            'x': x,
            'b': float(b),
            'y': y,
            'x_row': int(x_row),
            'y_row': int(y_row),
            'z': z
        }, body='int xid = get_global_id(0);\nz[z.offset + xid] = a * x[x.offset + (xid % x_row)] + b * y[y.offset + (xid % y_row)];')
        self.z = z

    def allocate(self, allocator: Allocator):
        _buffer = allocator.allocate(size=self.size)

        self.z.set_buffer_view(view=_buffer)

    def get_work_size(self) -> tuple[int, ...]:
        return (self.size, 1, 1)

class ElementwiseOperation(Operation):
    def __init__(self, x: Tensor, operation: str, y: Tensor, z: Tensor):
        self.size, x_row, y_row = _operation_layout(x, y)
        hash = _hash(operation)

        super().__init__(identifier=f'elemtwise_{hash}', arguments={
            'x': x,
            'y': y,
            'x_row': int(x_row),
            'y_row': int(y_row),
            'z': z
        }, body='int xid = get_global_id(0);\nz[z.offset + xid] = x[x.offset + (xid % x_row)] * y[y.offset + (xid % y_row)];')
        self.z = z

    def allocate(self, allocator: Allocator):
        _buffer = allocator.allocate(size=self.size)

        self.z.set_buffer_view(view=_buffer)

    def get_work_size(self) -> tuple[int, ...]:
        return (self.size, 1, 1)

class MATMULOperation(Operation):
    def __init__(self, a: Tensor, b: Tensor, z: Tensor):
        super().__init__(identifier='matmul', arguments={
            'a': a,
            'a_x': a.get_shape()[2],
            'b': b,
            'z': z
        }, body="""int xid = get_global_id(0);
            int yid = get_global_id(1);
            
            int xsize = get_global_size(0);
            int ysize = get_global_size(1);
            
            float result = 0.0;
            for (int i = 0; i < a_x; i++) {
                result += a[a.offset + yid * a_x + i] * b[b.offset + i * xsize + xid];
            }
            
            z[z.offset + yid * xsize + xid] = result;
            """
        )
        self.z = z

    def allocate(self, allocator: Allocator):
        _buffer = allocator.allocate(size=_shape_size(self.z.get_shape()))

        self.z.set_buffer_view(view=_buffer)

    def get_work_size(self) -> tuple[int, ...]:
        return self.z.get_shape()[::-1]