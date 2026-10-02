"""
EJEMPLO DE SÍNTESIS POR GUIADO DEL ACOPLADOR (dos trayectorias P y Q).

    uv run python examples/sintesis_movimiento.py

Se dan muchas posiciones sincronizadas de dos puntos P_i, Q_i de la puerta
(acoplador) y el programa devuelve la geometría que mejor la guía:
pivotes A y D, longitudes de las barras y P, Q en el marco del acoplador.

Para poder comprobar el resultado, las trayectorias se generan con un
mecanismo de referencia conocido; en un caso real vendrían del plano o de
una medición.

    Caso 1: datos exactos, pivotes libres.
    Caso 2: datos con ruido (±0.2 mm), pivotes A y D fijos.
    Caso 3: datos con ruido, pivotes libres.
    Caso 4: datos con ruido, cerrado y abierto con más peso.

Unidades mm - kg - s.
"""

import numpy as np

from fourbarhinge import (
    CouplerMotionSynthesis, FourBarDynamics, FourBarPlotter, LinkBounds,
    Point, SynthesisPlotter,
)
from fourbarhinge.synthesis import coupler_point_positions


def title(text: str) -> None:
    print(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


# ---------------------------------------------------------------------
# Mecanismo de referencia y trayectorias de P y Q
# ---------------------------------------------------------------------
reference = {
    "ax": 0.0, "ay": 0.0, "theta_ground": 0.0,
    "lg": 60.0, "li": 25.0, "lc": 70.0, "lo": 55.0,
    "u": 50.0, "v": 20.0,                 # P en el acoplador (local)
}
q_reference = Point(x=90.0, y=-10.0)     # Q en el acoplador (local)
branch = 1
n_poses = 30
theta = np.deg2rad(np.linspace(60.0, 150.0, n_poses))
track_p, _, _, _ = coupler_point_positions(reference, theta, branch)
track_q, _, _, _ = coupler_point_positions(
    dict(reference, u=q_reference.x, v=q_reference.y), theta, branch)

# Medición con ruido (p.ej. puntos leídos de un plano)
rng = np.random.default_rng(0)
noise = 0.2
noisy_p = track_p + rng.normal(0.0, noise, track_p.shape)
noisy_q = track_q + rng.normal(0.0, noise, track_q.shape)

bounds = LinkBounds(
    min_length=15.0,
    max_length=120.0,
    min_transmission_angle=np.deg2rad(20.0),
)
pivot_a = Point(x=0.0, y=0.0)
pivot_d = Point(x=60.0, y=0.0)


def compare(result) -> None:
    """Tabla resultado vs. referencia."""
    print(result.summary())
    values = dict(reference, uq=q_reference.x, vq=q_reference.y)
    found = dict(result.params, uq=result.second_point.x,
                 vq=result.second_point.y)
    print(f"\n  {'':>8}{'obtenido':>12}{'referencia':>12}")
    for name in ("lg", "li", "lc", "lo", "u", "v", "uq", "vq"):
        print(f"  {name:>8}{found[name]:12.3f}{values[name]:12.3f}")


def reference_error(track_p: np.ndarray, track_q: np.ndarray) -> float:
    """Error RMS del mecanismo de referencia frente a unos datos."""
    errors = np.concatenate((np.linalg.norm(track_p - noisy_p, axis=1),
                             np.linalg.norm(track_q - noisy_q, axis=1)))
    return float(np.sqrt(np.mean(errors**2)))


# ---------------------------------------------------------------------
# Caso 1: datos exactos, pivotes libres
# ---------------------------------------------------------------------
title(f"CASO 1: {n_poses} posiciones exactas, pivotes libres")
exact = CouplerMotionSynthesis(track_p, track_q, bounds).solve()
compare(exact)
print("\nCon más de 5 posiciones el movimiento del acoplador sólo lo"
      "\nreproduce el mecanismo original (salvo intercambiar entrada y"
      "\nsalida): se recupera exactamente.")

# ---------------------------------------------------------------------
# Caso 2: ruido, pivotes fijos
# ---------------------------------------------------------------------
title(f"CASO 2: ruido ±{noise} mm, pivotes A y D fijos")
fixed = CouplerMotionSynthesis(
    noisy_p, noisy_q, bounds, ground_pivot_a=pivot_a, ground_pivot_d=pivot_d,
).solve()
compare(fixed)

# ---------------------------------------------------------------------
# Caso 3: ruido, pivotes libres
# ---------------------------------------------------------------------
title(f"CASO 3: ruido ±{noise} mm, pivotes libres")
free = CouplerMotionSynthesis(noisy_p, noisy_q, bounds).solve()
compare(free)
door = np.unwrap(np.arctan2(*(track_q - track_p).T[::-1]))
print(f"\nError del mecanismo de referencia con estos datos: "
      f"{reference_error(track_p, track_q):.3f} mm"
      f"\nOtra geometría muy distinta ajusta igual de bien: con ruido y un"
      f"\ngiro de la puerta de sólo {np.rad2deg(np.ptp(door)):.0f}° los "
      f"datos no distinguen entre ellas."
      f"\nFijar los pivotes (caso 2) o acotar las longitudes elige la que"
      f"\ninteresa.")

# ---------------------------------------------------------------------
# Caso 4: ruido, posiciones extremas con más peso
# ---------------------------------------------------------------------
title("CASO 4: ruido, cerrado y abierto con peso 50")
weights = np.ones(n_poses)
weights[[0, -1]] = 50.0
weighted = CouplerMotionSynthesis(
    noisy_p, noisy_q, bounds, weights=weights,
    ground_pivot_a=pivot_a, ground_pivot_d=pivot_d,
).solve()
print(weighted.summary())
print("\nError en los extremos [mm]:     P cerrado  Q cerrado  P abierto"
      "  Q abierto")
for name, result in (("caso 2 (sin peso)", fixed), ("caso 4 (peso 50)",
                                                     weighted)):
    ends = [np.linalg.norm(points[i] - target[i])
            for i in (0, -1)
            for points, target in ((result.achieved_points, noisy_p),
                                   (result.achieved_points_q, noisy_q))]
    print(f"  {name:<28}" + "".join(f"{e:11.3f}" for e in ends))
print("No llegan a 0: con ruido |PQ| no es constante en los datos y una"
      "\npuerta rígida no puede pasar exactamente por ambos puntos.")

# ---------------------------------------------------------------------
# Informe PDF del caso 2
# ---------------------------------------------------------------------
geometry = fixed.to_geometry(
    linear_density=1.6e-4,
    coupler_mass=0.4,
    coupler_center_of_mass=fixed.coupler_point,
)
dynamics = FourBarDynamics(fixed.to_kinematics(geometry),
                           gravity=(0.0, -9810.0))
plotter = FourBarPlotter(dynamics, branch=fixed.branch,
                         coupler_point=fixed.coupler_point)
pdf = SynthesisPlotter(plotter).build_report(
    "output/sintesis_movimiento.pdf", fixed,
    title="Síntesis por guiado del acoplador (pivotes fijos)")
print(f"\nInforme: {pdf}")
