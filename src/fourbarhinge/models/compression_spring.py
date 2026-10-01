
from pydantic import BaseModel
from .position import Attachment


class CompressionSpring(BaseModel):
    free_length: float
    constant: float
    a: Attachment
    b: Attachment

