"""
EJEMPLO DE SÍNTESIS DE TRAYECTORIA (ingeniería inversa de la geometría).

    uv run python examples/sintesis_trayectoria.py      (~2 min)

Se da el recorrido del punto P donde se aplica la fuerza (puntos globales,
en orden) y el programa devuelve la geometría: pivotes, longitudes de las
barras y posición local de P en el acoplador.

Para poder comprobar el resultado, los puntos objetivo se generan con un
mecanismo de referencia conocido; en un caso real vendrían del plano.

    Caso 1: todo libre (pivotes, longitudes, P, ángulos de entrada).
    Caso 2: pivotes A y D fijos (impuestos por la carcasa).
    Caso 3: pivotes fijos + ángulo de entrada en cada punto (sincronización).
    Caso 4: pocos puntos (3) -> infinitas soluciones; los límites de
            longitud acotan la búsqueda a las dimensiones deseadas.

Unidades mm - kg - s.
"""

import numpy as np

from fourbarhinge import (
    CouplerPathSynthesis, FourBarDynamics, FourBarPlotter, LinkBounds,
    Point, SynthesisPlotter,
)
from fourbarhinge.synthesis.path_synthesis import coupler_point_positions


def title(text: str) -> None:
    print(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


# ---------------------------------------------------------------------
# Mecanismo de referencia y recorrido objetivo de P
# ---------------------------------------------------------------------
reference = {
    "ax": 0.0, "ay": 0.0, "theta_ground": 0.0,   # pivote A y orientación A->D
    "lg": 60.0, "li": 25.0, "lc": 70.0, "lo": 55.0,
    "u": 50.0, "v": 20.0,                         # P en el acoplador (local)
}
branch = 1
theta_reference = np.deg2rad(np.linspace(60.0, 150.0, 7))
targets, _, _, _ = coupler_point_positions(reference, theta_reference, branch)

title("RECORRIDO OBJETIVO DE P (global, mm)")
for k, (x, y) in enumerate(targets, start=1):
    print(f"  {k}: ({x:8.3f}, {y:8.3f})")

bounds = LinkBounds(
    min_length=15.0,
    max_length=120.0,
    min_transmission_angle=np.deg2rad(20.0),
)
pivot_a = Point(x=reference["ax"], y=reference["ay"])
pivot_d = Point(x=reference["ax"] + reference["lg"], y=reference["ay"])


def compare(result) -> None:
    """Tabla resultado vs. referencia."""
    print(result.summary())
    print(f"\n  {'':>14}{'obtenido':>12}{'referencia':>12}")
    for name in ("lg", "li", "lc", "lo", "u", "v"):
        print(f"  {name:>14}{result.params[name]:12.3f}"
              f"{reference[name]:12.3f}")


# ---------------------------------------------------------------------
# Caso 1: todo libre
# ---------------------------------------------------------------------
title("CASO 1: pivotes, longitudes y P libres")
free = CouplerPathSynthesis(targets, bounds).solve(n_starts=30, seed=0)
print(free.summary())
print("\nCon todo libre puede salir otro mecanismo distinto al de referencia"
      "\nque pasa por los mismos puntos (cognados de Roberts, etc.).")

# ---------------------------------------------------------------------
# Caso 2: pivotes fijos
# ---------------------------------------------------------------------
title("CASO 2: pivotes A y D fijos")
fixed_pivots = CouplerPathSynthesis(
    targets, bounds, ground_pivot_a=pivot_a, ground_pivot_d=pivot_d,
).solve(n_starts=30, seed=0)
compare(fixed_pivots)

# ---------------------------------------------------------------------
# Caso 3: pivotes fijos + ángulo de entrada en cada punto
# ---------------------------------------------------------------------
title("CASO 3: pivotes fijos + ángulos de entrada prescritos")
timed = CouplerPathSynthesis(
    targets, bounds, ground_pivot_a=pivot_a, ground_pivot_d=pivot_d,
    theta_inputs=theta_reference, branches=(branch,),
).solve(n_starts=30, seed=0)
compare(timed)

# ---------------------------------------------------------------------
# Caso 4: sólo 3 puntos (inicio, medio, final) y longitudes acotadas
# ---------------------------------------------------------------------
# Con menos de 5 puntos hay infinitas soluciones. Unos límites de
# longitud estrechos alrededor de las dimensiones deseadas hacen que la
# solución devuelta sea parecida a lo que se busca.
title("CASO 4: 3 puntos, longitudes entre 20 y 75 mm")
three = targets[[0, 3, 6]]
bounded = CouplerPathSynthesis(
    three,
    LinkBounds(min_length=20.0, max_length=75.0,
               min_transmission_angle=np.deg2rad(20.0)),
    ground_pivot_a=pivot_a, ground_pivot_d=pivot_d,
).solve(n_starts=30, seed=0)
compare(bounded)
print("\nPasa exactamente por los 3 puntos, pero no es el mecanismo de"
      "\nreferencia: con pocos puntos se devuelve una de las soluciones."
      "\nConviene comprobar la trayectoria completa entre los puntos.")

# ---------------------------------------------------------------------
# Informe PDF del caso 2
# ---------------------------------------------------------------------
geometry = fixed_pivots.to_geometry(
    linear_density=1.6e-4,
    coupler_mass=0.4,
    coupler_center_of_mass=fixed_pivots.coupler_point,
)
dynamics = FourBarDynamics(fixed_pivots.to_kinematics(geometry),
                           gravity=(0.0, -9810.0))
plotter = FourBarPlotter(dynamics, branch=fixed_pivots.branch,
                         coupler_point=fixed_pivots.coupler_point)
pdf = SynthesisPlotter(plotter).build_report(
    "output/sintesis_trayectoria.pdf", fixed_pivots,
    title="Síntesis de trayectoria (pivotes fijos)")
print(f"\nInforme: {pdf}")
