
"""
EJEMPLO DE SÍNTESIS: de un recorrido deseado a la geometría y los muelles.

    uv run python examples/sintesis_bisagra.py

1. Recorrido deseado de un punto P de la puerta (acoplador).
2. CouplerPathSynthesis busca la geometría con los pivotes fijos dados.
3. Se construye la dinámica de la geometría obtenida (masas).
4. EffortTravelCurve: esfuerzo-recorrido sin muelles.
5. SpringSynthesis: posiciona/dimensiona un muelle de compresión y uno de
   torsión para acercarse a una curva esfuerzo-recorrido objetivo.
6. Informe PDF A4 (síntesis + análisis dinámico completo).

Unidades mm - kg - s  ->  fuerza mN, par mN·mm, energía µJ.
"""

import numpy as np

from fourbarhinge import (
    Actuator, BarId, CompressionSpringDesign, CouplerPathSynthesis,
    EffortTravelCurve, FourBarDynamics, FourBarPlotter, LinkBounds, Point,
    ReturnSimulation, SpringSynthesis, SynthesisPlotter, TorsionSpringDesign,
)

# ---------------------------------------------------------------------
# 1. Recorrido deseado del punto P (global, mm), en orden: cerrado -> abierto
# ---------------------------------------------------------------------
targets = np.array([
    [90.0, 10.0],
    [92.0, 25.0],
    [88.0, 40.0],
    [78.0, 54.0],
    [62.0, 64.0],
    [42.0, 70.0],
    [22.0, 70.0],
])

# ---------------------------------------------------------------------
# 2. Síntesis de la geometría
# ---------------------------------------------------------------------
# Los pivotes fijos A y D vienen impuestos por la carcasa del mueble.
# Si no se conocen, omitirlos: también se buscan (más incógnitas).
synthesis = CouplerPathSynthesis(
    targets,
    LinkBounds(
        min_length=15.0,
        max_length=120.0,
        min_transmission_angle=np.deg2rad(30.0),   # evita bloqueos
    ),
    ground_pivot_a=Point(x=0.0, y=0.0),
    ground_pivot_d=Point(x=40.0, y=-10.0),
)
path_result = synthesis.solve(n_starts=60, seed=0)
print("GEOMETRÍA\n" + path_result.summary())
if not path_result.feasible:
    raise SystemExit("La geometría no es montable en todo el recorrido")

# ---------------------------------------------------------------------
# 3. Dinámica de la geometría obtenida
# ---------------------------------------------------------------------
# Barras de acero de 10x2 mm ≈ 1.6e-4 kg/mm; la puerta pesa 0.4 kg
# y su centro de masas está cerca del punto P.
geometry = path_result.to_geometry(
    linear_density=1.6e-4,
    coupler_mass=0.4,
    coupler_center_of_mass=path_result.coupler_point,
    coupler_inertia=0.4 * 120.0**2 / 12.0,
)
kinematics = path_result.to_kinematics(geometry)
dynamics = FourBarDynamics(kinematics, gravity=(0.0, -9810.0))

# ---------------------------------------------------------------------
# 4. Curva esfuerzo-recorrido sin muelles
# ---------------------------------------------------------------------
curve = EffortTravelCurve(
    dynamics,
    coupler_point=path_result.coupler_point,
    theta_start=path_result.theta_start,
    theta_end=path_result.theta_end,
    branch=path_result.branch,
    samples=150,
)
no_springs = curve.effort([], [])
print(f"\nSin muelles: F = {np.nanmin(no_springs):.0f} … "
      f"{np.nanmax(no_springs):.0f} mN, recorrido = "
      f"{curve.travel[-1]:.1f} mm")

# ---------------------------------------------------------------------
# 5. Muelles para una curva objetivo
# ---------------------------------------------------------------------
# Objetivo "retención en ambos extremos": al principio cuesta abrir
# (+3 N) y al final la puerta se mantiene abierta (-2 N).
travel_total = curve.travel[-1]
target_travel = np.array([0.0, travel_total])
target_effort = np.array([3000.0, -2000.0])

compression = CompressionSpringDesign(
    bar_a=BarId.GROUND, bar_b=BarId.OUTPUT,
    a_x=(0.0, 40.0), a_y=(-15.0, 15.0),      # sobre la carcasa
    b_x=(10.0, 100.0), b_y=(-5.0, 5.0),      # sobre la barra de salida
    constant=(50.0, 2000.0),                 # mN/mm
    free_length=(20.0, 120.0),               # mm
)
torsion = TorsionSpringDesign(
    bar_a=BarId.GROUND, bar_b=BarId.INPUT,   # en la bisagra A
    constant=(0.0, 200_000.0),               # mN·mm/rad
    free_angle=(-np.pi, np.pi),
)
spring_synthesis = SpringSynthesis(
    curve, target_travel, target_effort,
    compression_designs=[compression],
    torsion_designs=[torsion],
    min_spring_length=15.0,                  # longitud a bloque
)
spring_result = spring_synthesis.solve(n_starts=20, seed=0)
print("\nMUELLES\n" + spring_result.summary())

# ---------------------------------------------------------------------
# 6. Informe: síntesis + análisis dinámico con los muelles obtenidos
# ---------------------------------------------------------------------
dynamics.compression_springs = spring_result.compression_springs
dynamics.torsion_springs = spring_result.torsion_springs
push = Actuator(bar=BarId.COUPLER, point=path_result.coupler_point,
                direction=(1.0, 0.0))
plotter = FourBarPlotter(dynamics, branch=path_result.branch, actuator=push,
                         coupler_point=path_result.coupler_point)

synthesis_pdf = SynthesisPlotter(plotter).build_report(
    "output/sintesis_bisagra.pdf", path_result, spring_result)
print(f"\nInforme de síntesis: {synthesis_pdf}")

# Retorno libre hacia cerrado: se suelta donde el esfuerzo aún es > 0
# (zona de autocierre). Desde la zona F < 0 la puerta se queda abierta.
closing = np.where(spring_result.effort > 0.0)[0]
release = curve.theta_input[closing[-1] - 5]
print(f"Se suelta en el recorrido "
      f"{curve.travel[closing[-1] - 5]:.1f} mm (θ = "
      f"{np.rad2deg(release):.1f}°)")
trajectory = ReturnSimulation(dynamics, path_result.branch).run(
    release, path_result.theta_start, dt=1e-4, t_max=0.5)
print(f"Retorno: {trajectory.stop_reason}")
analysis_pdf = plotter.build_report("output/sintesis_dinamica.pdf",
                                    trajectory)
print(f"Informe dinámico: {analysis_pdf}")
