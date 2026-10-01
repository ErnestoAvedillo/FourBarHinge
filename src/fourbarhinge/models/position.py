
from pydantic import BaseModel
import numpy as np
from math import cos, sin


class Position(BaseModel):
    length: float
    angle: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.length * cos(self.angle),
            self.length * sin(self.angle),
            ])


class Attachment(BaseModel):
    bar: int
    position: Position

