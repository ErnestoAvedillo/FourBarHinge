"""
EJEMPLO DE SÍNTESIS DE MUELLES (esfuerzo-recorrido -> posición del muelle).

    uv run python examples/sintesis_muelles.py

Con una geometría ya conocida, se da la curva esfuerzo-recorrido deseada
en el punto P donde se aplica la fuerza y el programa busca dónde anclar
el muelle y cómo dimensionarlo.

El esfuerzo es la fuerza estática en P tangente a su trayectoria:
    F > 0: hay que empujar en el sentido del recorrido.
    F < 0: la puerta avanza sola (hay que frenarla).

    Caso 1: muelle de catálogo (k y L0 fijos) -> sólo se buscan anclajes.
            La curva objetivo se genera con un muelle conocido para
            comprobar que se recupera.
    Caso 2: curva de diseño con un muelle de compresión libre
            (anclajes, k y L0): no basta.
    Caso 3: misma curva con compresión + torsión, dando más peso a los
            extremos.
    Caso 4: la fuerza no es tangente: se apoya en un bulón fijo S y actúa
            en la línea S -> P (cilindro, amortiguador, tirador...).

Unidades mm - kg - s  ->  fuerza mN, par mN·mm.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fourbarhinge import (
    Bar, BarId, CompressionSpringDesign, EffortTravelCurve,
    FourBarDynamics, FourBarGeometry, FourBarKinematics, FourBarPlotter,
    Point, SpringSynthesis, SynthesisPlotter, TorsionSpringDesign,
)


def title(text: str) -> None:
    print(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


def bar(length: float, mass: float) -> Bar:
    return Bar(length=length, mass=mass, inertia=mass * length**2 / 12.0,
               center_of_mass=Point(x=length / 2.0, y=0.0))


# ---------------------------------------------------------------------
# Geometría conocida (p.ej. la obtenida en sintesis_trayectoria.py)
# ---------------------------------------------------------------------
# El acoplador es la puerta: 0.4 kg con el centro de masas en P.
coupler_point = Point(x=50.0, y=20.0)
geometry = FourBarGeometry(
    position_ground=Point(x=0.0, y=0.0),
    bar={
        BarId.GROUND: bar(60.0, 0.05),
        BarId.INPUT: bar(25.0, 0.02),
        BarId.COUPLER: Bar(length=70.0, mass=0.4,
                           inertia=0.4 * 120.0**2 / 12.0,
                           center_of_mass=coupler_point),
        BarId.OUTPUT: bar(55.0, 0.04),
    },
)
branch = 1
theta_start, theta_end = np.deg2rad(60.0), np.deg2rad(150.0)
kinematics = FourBarKinematics(geometry, theta_ground=0.0)
dynamics = FourBarDynamics(kinematics, gravity=(0.0, -9810.0))

curve = EffortTravelCurve(dynamics, coupler_point, theta_start, theta_end,
                          branch=branch, samples=150)
travel_total = curve.travel[-1]

title("SIN MUELLES")
no_springs = curve.effort([], [])
print(f"Recorrido de P = {travel_total:.1f} mm")
for s in np.linspace(0.0, travel_total, 6):
    print(f"  s = {s:6.1f} mm   F = {np.interp(s, curve.travel, no_springs):8.0f} mN")

# Zona donde se permite anclar el muelle (coordenadas locales de cada barra)
anchor_zone = dict(
    bar_a=BarId.GROUND, bar_b=BarId.INPUT,
    a_x=(-20.0, 60.0), a_y=(-30.0, 20.0),   # sobre la carcasa
    b_x=(5.0, 25.0), b_y=(-5.0, 5.0),       # sobre la barra de entrada
)

# ---------------------------------------------------------------------
# Caso 1: muelle de catálogo, sólo posición
# ---------------------------------------------------------------------
# Para comprobar el método, la curva objetivo se genera con un muelle de
# referencia conocido; la síntesis debe encontrar sus anclajes.
title("CASO 1: muelle de catálogo (k = 300 mN/mm, L0 = 50 mm)")
stock = CompressionSpringDesign(**anchor_zone,
                                constant=(300.0, 300.0),     # low == high:
                                free_length=(50.0, 50.0))    # valor fijo
reference_spring = stock.build({"a_x": 0.0, "a_y": -20.0, "b_x": 20.0,
                                "b_y": 0.0, "constant": 300.0,
                                "free_length": 50.0})
# Curva completa: el objetivo se interpola linealmente, así que con
# pocos puntos se pierde la forma (y el error ya no puede ser 0).
stock_travel = curve.travel
stock_effort = curve.effort([reference_spring], [])
print("Objetivo (muestra): " + ", ".join(
    f"({s:.1f}, {f:.0f})"
    for s, f in zip(stock_travel[::30], stock_effort[::30])))

stock_result = SpringSynthesis(
    curve, stock_travel, stock_effort,
    compression_designs=[stock], min_spring_length=15.0,
).solve(n_starts=20, seed=0)
print(stock_result.summary())
spring = stock_result.compression_springs[0]
a_xy, b_xy = spring.a.position.to_array(), spring.b.position.to_array()
print(f"Anclaje en carcasa (local): ({a_xy[0]:6.2f}, {a_xy[1]:6.2f}) mm"
      f"   referencia (0, -20)\n"
      f"Anclaje en entrada (local): ({b_xy[0]:6.2f}, {b_xy[1]:6.2f}) mm"
      f"   referencia (20, 0)")
print("Puede haber otros anclajes que den la misma curva; si el error es"
      "\n~0, cualquiera de ellos sirve.")

# ---------------------------------------------------------------------
# Curva objetivo de diseño: retención en cerrado, frena al final
# ---------------------------------------------------------------------
target_travel = np.array([0.0, 0.5, 1.0]) * travel_total
target_effort = np.array([2000.0, 0.0, -1000.0])
title("CURVA OBJETIVO DE DISEÑO")
print(", ".join(f"({s:.1f} mm, {f:.0f} mN)"
                for s, f in zip(target_travel, target_effort)))

# ---------------------------------------------------------------------
# Caso 2: muelle de compresión libre
# ---------------------------------------------------------------------
title("CASO 2: muelle de compresión libre")
free = CompressionSpringDesign(**anchor_zone,
                               constant=(20.0, 2000.0),
                               free_length=(20.0, 120.0))
free_result = SpringSynthesis(
    curve, target_travel, target_effort,
    compression_designs=[free], min_spring_length=15.0,
).solve(n_starts=20, seed=0)
print(free_result.summary())
print("Un solo muelle de compresión no puede seguir esta curva: el"
      "\noptimizador acaba en los límites de k y L0. Hace falta otro muelle.")

# ---------------------------------------------------------------------
# Caso 3: compresión + torsión en A, con más peso en los extremos
# ---------------------------------------------------------------------
title("CASO 3: compresión + torsión, extremos con peso 5")
torsion = TorsionSpringDesign(bar_a=BarId.GROUND, bar_b=BarId.INPUT,
                              constant=(0.0, 200_000.0),
                              free_angle=(-np.pi, np.pi))
both_result = SpringSynthesis(
    curve, target_travel, target_effort,
    compression_designs=[free], torsion_designs=[torsion],
    weights=np.array([5.0, 1.0, 5.0]),
    min_spring_length=15.0,
).solve(n_starts=20, seed=0)
print(both_result.summary())

# ---------------------------------------------------------------------
# Comparación y comprobación con la dinámica completa
# ---------------------------------------------------------------------
title("COMPARACIÓN CON LA CURVA DE DISEÑO [mN]")
print(f"  {'s [mm]':>8}{'objetivo':>10}{'caso 2':>10}{'caso 3':>10}")
for s in np.linspace(0.0, travel_total, 7):
    row = [np.interp(s, target_travel, target_effort)] + [
        np.interp(s, curve.travel, r.effort)
        for r in (free_result, both_result)]
    print(f"  {s:8.1f}" + "".join(f"{v:10.0f}" for v in row))

# El esfuerzo de EffortTravelCurve coincide con resolver el equilibrio
# con FourBarDynamics y una fuerza tangente a la trayectoria de P.
dynamics.compression_springs = both_result.compression_springs
dynamics.torsion_springs = both_result.torsion_springs
k = len(curve.travel) // 2
kinematics.solve_configuration(curve.theta_input[k], branch)
kinematics.solve_theta_dot(0.0)
push = curve.actuator(k)          # fuerza tangente en la muestra k
check = dynamics.solve_required_effort(push, theta_input_ddot=0.0)
print(f"\nComprobación en s = {curve.travel[k]:.1f} mm: curva = "
      f"{both_result.effort[k]:.1f} mN, dinámica = {check:.1f} mN")
dynamics.compression_springs, dynamics.torsion_springs = [], []

# ---------------------------------------------------------------------
# Caso 4: fuerza apoyada en un bulón fijo S
# ---------------------------------------------------------------------
# La fuerza no es tangente: la ejerce p.ej. un cilindro o amortiguador
# articulado en el bulón S (global) y en P. Actúa siempre en la línea
# S -> P y el recorrido es la carrera |SP| - |SP|₀.
#     F > 0: empuja (aleja P de S)   F < 0: tira (acerca P a S)
title("CASO 4: fuerza apoyada en el bulón S = (80, -40)")
bolt = Point(x=80.0, y=-40.0)
bolt_curve = EffortTravelCurve(dynamics, coupler_point, theta_start,
                               theta_end, branch=branch, samples=150,
                               force_origin=bolt)
stroke = bolt_curve.travel
print(f"|SP| = {bolt_curve.force_length[0]:.1f} -> "
      f"{bolt_curve.force_length[-1]:.1f} mm  (carrera "
      f"{stroke[-1]:+.1f} mm)")
bolt_no_springs = bolt_curve.effort([], [])
for i in np.linspace(0, len(stroke) - 1, 6).astype(int):
    print(f"  carrera = {stroke[i]:6.1f} mm   F = {bolt_no_springs[i]:8.0f} mN")

# Objetivo: el cilindro empuja 1.5 N al empezar y nada al final
bolt_target_travel = np.array([stroke[0], stroke[-1]])
bolt_target_effort = np.array([1500.0, 0.0])
# Ángulo libre acotado para que la deflexión no llegue a ±180° (donde
# el modelo la envuelve y el par cambia de signo)
bolt_torsion = TorsionSpringDesign(bar_a=BarId.GROUND, bar_b=BarId.INPUT,
                                   constant=(0.0, 200_000.0),
                                   free_angle=(0.0, np.pi))
bolt_result = SpringSynthesis(
    bolt_curve, bolt_target_travel, bolt_target_effort,
    compression_designs=[free], torsion_designs=[bolt_torsion],
    min_spring_length=15.0,
).solve(n_starts=20, seed=0)
print("\n" + bolt_result.summary())
for i in np.linspace(0, len(stroke) - 1, 6).astype(int):
    print(f"  carrera = {stroke[i]:6.1f} mm   objetivo = "
          f"{np.interp(stroke[i], bolt_target_travel, bolt_target_effort):6.0f}"
          f"   obtenido = {bolt_result.effort[i]:6.0f} mN")

# Comprobación con la dinámica: misma fuerza con la dirección S -> P
dynamics.compression_springs = bolt_result.compression_springs
dynamics.torsion_springs = bolt_result.torsion_springs
kinematics.solve_configuration(bolt_curve.theta_input[k], branch)
kinematics.solve_theta_dot(0.0)
check = dynamics.solve_required_effort(bolt_curve.actuator(k))
print(f"\nComprobación en carrera = {stroke[k]:.1f} mm: curva = "
      f"{bolt_result.effort[k]:.1f} mN, dinámica = {check:.1f} mN")

# ---------------------------------------------------------------------
# Informe PDF: una página esfuerzo-recorrido por caso
# ---------------------------------------------------------------------
plotter = FourBarPlotter(dynamics, branch=branch, actuator=push,
                         coupler_point=coupler_point)
synthesis_plotter = SynthesisPlotter(plotter)
pdf_path = Path("output/sintesis_muelles.pdf")
pdf_path.parent.mkdir(parents=True, exist_ok=True)
with PdfPages(pdf_path) as pdf:
    for result in (stock_result, free_result, both_result, bolt_result):
        fig = synthesis_plotter.page_springs(result)
        pdf.savefig(fig)
        plt.close(fig)
print(f"\nInforme: {pdf_path}")
