
from pydantic import BaseModel
from .barra import Point
from .constants import BarId


class ExternalForce(BaseModel):
    """Fuerza conocida aplicada en un punto de una barra.

    point: coordenadas locales de la barra.
    force: componentes (Fx, Fy) en ejes globales.
    torque: par puro adicional sobre la barra (eje z global).
    """
    bar: BarId
    point: Point
    force: tuple[float, float] = (0.0, 0.0)
    torque: float = 0.0


class Actuator(BaseModel):
    """Esfuerzo de magnitud desconocida λ que se quiere calcular.

    direction=None  -> par λ sobre `bar` (p.ej. motor/mano en la bisagra A).
    direction=(ux, uy) -> fuerza λ·u (se normaliza) aplicada en
    `point` (coordenadas locales de `bar`).
    local_direction=True -> u en ejes locales de la barra (gira con ella,
    p.ej. empuje siempre perpendicular al acoplador); False -> ejes globales.
    """
    bar: BarId
    point: Point = Point(x=0.0, y=0.0)
    direction: tuple[float, float] | None = None
    local_direction: bool = False
