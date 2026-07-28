from numbers import Number
from textwrap import indent
from typing import Any

from staticml.buffer import Buffer
from staticml.dtype import DType, float32


class OperationArg:
    def __init__(self, name: str, value: Any, dtype: DType = float32):
        self.name = name
        self.value = value
        self.dtype = dtype

        self._source_line = self.gen_source_line()
        self._call_line = self.gen_call_line()

    def gen_source_line(self) -> str:
        return f'const {self._dtype.c_name} {self._name}'

    def gen_call_line(self) -> str:
        return f'{self._dtype.cast(o=value)}'

    @property
    def source_line(self) -> str:
        return self._source_line

    @property
    def call_line(self) -> str:
        return self._call_line

class OperationBufferArg(OperationArg):
    def __init__(self, name, buffer: Buffer, dtype):
        super().__init__(name=name, value=buffer, dtype=dtype)

    def gen_source_line(self) -> str:
        return f'{self.value.asq.value} {self.dtype.c_name}* {self.name}'

    def gen_call_line(self) -> str:
        return f'({self.value.asq.value} {self.dtype.c_name}*) {self.value.name}'

class Operation:
    def __init__(
            self,
            name: str,
            args: list[OperationArg] | None = None,
            body: list[str] | None = None
    ):
        self.name = name
        self.args = args or {}
        self.body = body or []

        self._source_string = self.gen_source_string()
        self._call_string = self.gen_call_string()

    def gen_source_string(self) -> str:
        arg_strings = [arg.source_line for arg in self.args]

        return '\n'.join((
            f'void {self.name}_{hash(self):0x}(',
            indent(',\n'.join(arg_strings), '\t'),
            ') {',
            indent('\n'.join(self.body), '\t'),
            '}'
        ))

    def gen_call_string(self) -> str:
        arg_strings = [arg.call_line for arg in self.args]

        return '\n'.join((
            f'void {self.name}_{hash(self):0x}(',
            indent(',\n'.join(arg_strings), '\t'),
            ');'
        ))

    @property
    def source_string(self) -> str:
        return self._source_string

    @property
    def call_string(self) -> str:
        return self._call_string