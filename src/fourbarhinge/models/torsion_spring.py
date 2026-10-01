
from pydantic import BaseModel


class TorsionSpring(BaseModel):
    free_angle: float
    constant: float
    bar_a: int
    bar_b: int

