class Shape:
    def __init__(self, x: int = -1, y: int = -1, z: int = -1):
        self.x = x
        self.y = y
        self.z = z

        if self.z != -1:
            self.x = self.x if self.x != -1 else 1
            self.y = self.y if self.y != -1 else 1

        if self.y != -1:
            self.x = self.x if self.x != -1 else 1

        if all((self.x == -1, self.y == -1, self.z == -1)):
            self.x = 0

    def __repr__(self):
        return f"Shape({', '.join(f'{s}={v}' for s, v in [
            ('x', self.x),
            ('y', self.y),
            ('z', self.z)
        ] if v != -1)})"

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type int, not '{type(index)}'")

        if index < 0 or index > 2:
            raise ValueError(f"Index must be in the range of 0 to 2, not '{index}'")

        return (self.x, self.y, self.z)[index]

    def as_tuple(self, reverse: bool = False):
        if reverse:
            return tuple(x for x in (self.z, self.y, self.x) if x != -1)
        return tuple(x for x in (self.x, self.y, self.z) if x != -1)

    def copy(self) -> Shape:
        return Shape(x=self.x, y=self.y, z=self.z)

    @property
    def inflated(self) -> Shape:
        return Shape(
            x=self.x if self.x != -1 else 1,
            y=self.y if self.y != -1 else 1,
            z=self.z if self.z != -1 else 1
        )

    @staticmethod
    def from_numpy_shape(shape: tuple[int, ...]) -> Shape:
        if len(shape) > 3:
            raise ValueError("Numpy shape has more than 3 dimensions")

        return Shape(*shape[::-1])