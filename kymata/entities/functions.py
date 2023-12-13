from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class Function:
    name: str
    values: NDArray
    tstep: float

    def downsampled(self, rate: int):
        return Function(
            name=self.name,
            values=self.values[:, ::rate],
            tstep=self.tstep * rate
        )
