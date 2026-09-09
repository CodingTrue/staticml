from textwrap import indent

def _join_list(a, delimiter = ', '):
    return delimiter.join(str(x) for x in a)

def _indent_list(a, delimiter = ', '):
    return indent(_join_list(a, delimiter), '\t')