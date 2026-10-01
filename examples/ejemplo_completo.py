
"""
EJEMPLO COMPLETO: bisagra de 4 barras para una puerta de mueble.

Ejecutar desde la raíz del proyecto:
    uv run python examples/ejemplo_completo.py

Flujo:
    1. Geometría y masas
    2. Muelles y cargas externas conocidas
    3. Cinemática (posición, velocidad, aceleración)
    4. Dinámica reducida a 1 DOF: M, C, G, Q  ->  ẗi
    5. Esfuerzo necesario (par en A o fuerza sobre el acoplador)
    6. Reacciones en las articulaciones
    7. Barrido estático: curva de esfuerzo
    8. Simulación del retorno libre hasta home
    9. Informe PDF A4

UNIDADES: el código no convierte nada; deben ser coherentes.
Aquí se usa mm - kg - s, por lo que:
    fuerza  = kg·mm/s²   = mN
    par     = mN·mm
    energía = kg·mm²/s²  = µJ
    g       = 9810 mm/s²
Con m - kg - s (SI) bastaría con g = 9.81 y fuerzas en N.

CONVENIO DE ÁNGULOS:
    Todos los ángulos (ti, tc, to) son absolutos respecto a la barra fija
    (A -> D), en radianes, antihorarios. La barra de salida se mide de
    D hacia C.

         B ●────────────● C
          /   acoplador  \
     entrada            salida
        /                  \
     A ●────────────────────● D
              fija
"""

import numpy as np

from fourbarhinge.models.barra import Bar, Point, FourBarGeometry
from fourbarhinge.models.constants import BarId
from fourbarhinge.models.position import Position, Attachment
from fourbarhinge.models.compression_spring import CompressionSpring
from fourbarhinge.models.torsion_spring import TorsionSpring
from fourbarhinge.models.external_force import ExternalForce, Actuator
from fourbarhinge.kinematics.kinematics import FourBarKinematics
from fourbarhinge.dynamics.dynamics import FourBarDynamics
from fourbarhinge.dynamics.reactions import FourBarReactions
from fourbarhinge.simulation.return_simulation import ReturnSimulation
from fourbarhinge.plots.plots import FourBarPlotter, Units


def title(text: str) -> None:
    print(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


def deg(rad: float) -> str:
    return f"{np.rad2deg(rad):8.2f}°"


# =====================================================================
# 1. GEOMETRÍA Y MASAS
# =====================================================================
#
# Cada barra se define en su marco local:
#   origen = articulación inicial (A, B o D), eje x hacia la final.
# center_of_mass: COM en ese marco local.
# inertia: momento de inercia Izz respecto al COM (kg·mm²).
# Para una barra uniforme: I = m·L²/12, COM = (L/2, 0).

def uniform_bar(length: float, mass: float) -> Bar:
    return Bar(
        length=length,
        mass=mass,
        inertia=mass * length**2 / 12.0,
        center_of_mass=Point(x=length / 2.0, y=0.0),
    )


geometry = FourBarGeometry(
    # Posición global del pivote A
    position_ground=Point(x=0.0, y=0.0),
    bar={
        BarId.GROUND: uniform_bar(60.0, 0.05),   # A-D (carcasa)
        BarId.INPUT: uniform_bar(25.0, 0.02),    # A-B
        # El acoplador lleva la puerta: COM desplazado fuera de la línea BC
        BarId.COUPLER: Bar(length=70.0, mass=0.06, inertia=35.0,
                           center_of_mass=Point(x=40.0, y=5.0)),
        BarId.OUTPUT: uniform_bar(55.0, 0.04),   # D-C
    },
)

# Orientación global de la barra fija (0 = horizontal).
THETA_GROUND = np.deg2rad(0.0)

# Rama de montaje: +1 o -1 elige una de las dos soluciones del cierre.
# Comprobar en el dibujo del informe que corresponde a la bisagra real.
BRANCH = -1

kinematics = FourBarKinematics(geometry=geometry, theta_ground=THETA_GROUND)


# =====================================================================
# 2. MUELLES Y CARGAS EXTERNAS CONOCIDAS
# =====================================================================
#
# Muelle de COMPRESIÓN entre dos puntos de dos barras.
#   Attachment.position está en polares (longitud, ángulo) en el marco
#   local de la barra. Sólo empuja: si se estira más allá de
#   free_length, su fuerza es 0.
compression_springs = [
    CompressionSpring(
        free_length=60.0,          # mm
        constant=500.0,            # mN/mm  (= 0.5 N/mm)
        a=Attachment(bar=BarId.GROUND,
                     position=Position(length=0.0, angle=0.0)),   # en A
        b=Attachment(bar=BarId.COUPLER,
                     position=Position(length=50.0, angle=0.0)),
    ),
]

# Muelle de TORSIÓN en una bisagra: par = -k·(φ - φ0),
#   φ = ángulo(bar_b) - ángulo(bar_a).
# (GROUND, INPUT) -> bisagra A; (INPUT, COUPLER) -> B; etc.
torsion_springs = [
    TorsionSpring(
        free_angle=np.deg2rad(20.0),   # posición de reposo del muelle
        constant=50_000.0,             # mN·mm/rad (= 0.05 N·m/rad)
        bar_a=BarId.GROUND,
        bar_b=BarId.INPUT,
    ),
]

# Cargas externas CONOCIDAS (siempre presentes, p.ej. un tirador o un
# accesorio colgado). force en ejes globales, point en ejes locales.
external_forces = [
    ExternalForce(
        bar=BarId.COUPLER,
        point=Point(x=70.0, y=0.0),    # extremo del acoplador (C)
        force=(0.0, -200.0),           # 200 mN hacia abajo
    ),
]

dynamics = FourBarDynamics(
    kinematics,
    compression_springs=compression_springs,
    torsion_springs=torsion_springs,
    external_forces=external_forces,
    gravity=(0.0, -9810.0),            # mm/s², ejes globales
)


# =====================================================================
# 3. CINEMÁTICA
# =====================================================================
#
# Orden obligatorio: posición -> velocidad -> aceleración.

title("3. CINEMÁTICA")

theta_input = np.deg2rad(100.0)
theta_input_dot = -20.0          # rad/s
theta_input_ddot = 0.0           # rad/s²

kinematics.solve_configuration(theta_input, BRANCH)
# Si el ángulo no es montable, solve_configuration devuelve una posición
# falsa: comprobarlo siempre.
assert kinematics.get_closure_error() < 1e-6, "Posición no montable"

kinematics.solve_theta_dot(theta_input_dot)
kinematics.solve_theta_ddot(theta_input_ddot)

theta = kinematics.configuration.theta
theta_dot = kinematics.theta_dot.theta_dot
theta_ddot = kinematics.theta_ddot.theta_ddot
for bar in (BarId.INPUT, BarId.COUPLER, BarId.OUTPUT):
    print(f"{bar.name:8s} θ = {deg(theta[bar])}   "
          f"ω = {theta_dot[bar]:8.3f} rad/s   "
          f"α = {theta_ddot[bar]:10.2f} rad/s²")

print("\nArticulaciones (global, mm):")
for name, p in kinematics.get_joint_positions().items():
    print(f"  {name}: ({p[0]:7.2f}, {p[1]:7.2f})")

# Cualquier punto de una barra: posición, velocidad y aceleración globales
handle = Point(x=70.0, y=0.0)
print("\nExtremo del acoplador:")
print("  posición    :", kinematics.get_local_to_global(BarId.COUPLER, handle))
print("  velocidad   :", kinematics.get_point_velocity(BarId.COUPLER, handle))
print("  aceleración :",
      kinematics.get_point_acceleration(BarId.COUPLER, handle))

# Relación de velocidades Γ = [1, dtc/dti, dto/dti]
print("\nΓ (relación de velocidades):", kinematics.get_velocity_ratios())


# =====================================================================
# 4. DINÁMICA REDUCIDA A 1 DOF
# =====================================================================
#
#   M_red·ẗi + C_red + G_red = Q_muelles + Q_externas + λ·Q_actuador
#
# Todos los términos son "pares equivalentes" referidos a la barra de
# entrada (mN·mm).

title("4. DINÁMICA REDUCIDA")

print("M_MR (cadena abierta, coords. relativas):\n",
      dynamics.get_mass_matrix())
print("M_fb (coords. absolutas ti, tc, to):\n",
      dynamics.get_mass_matrix_fourbar())
print(f"M_red       = {dynamics.get_reduced_mass():12.3f} kg·mm²")
print(f"C_red       = {dynamics.get_reduced_coriolis():12.3f} mN·mm")
print(f"G_red       = {dynamics.get_reduced_gravity():12.3f} mN·mm")
print(f"Q_muelles   = {dynamics.get_reduced_spring_force():12.3f} mN·mm")
print(f"Q_externas  = {dynamics.get_reduced_external_force():12.3f} mN·mm")

# Dinámica directa: ¿qué aceleración tiene si se suelta aquí?
# (ojo: actualiza kinematics.theta_ddot)
free_ddot = dynamics.solve_theta_input_ddot()
print(f"\nSi se suelta a {theta_input_dot} rad/s -> ẗi = {free_ddot:.2f} "
      "rad/s²")

print(f"\nEnergía cinética          = "
      f"{dynamics.get_kinetic_energy_direct():10.2f} µJ")
print(f"Energía de los muelles    = "
      f"{dynamics.get_spring_potential_energy():10.2f} µJ")


# =====================================================================
# 5. ESFUERZO NECESARIO
# =====================================================================
#
# Actuator = esfuerzo de magnitud desconocida λ.
#   direction=None              -> par sobre la barra
#   direction=(ux, uy)          -> fuerza en esa dirección (global)
#   local_direction=True        -> la dirección gira con la barra
# solve_required_effort(actuator, ẗi) devuelve λ.
#   λ > 0: en el sentido indicado; λ < 0: en el contrario.

title("5. ESFUERZO NECESARIO (estático en θ = 100°)")

# Estado estático: velocidad 0 y aceleración 0
kinematics.solve_configuration(theta_input, BRANCH)
kinematics.solve_theta_dot(0.0)

# a) Par en la bisagra A (motor o mano en el eje)
torque_A = Actuator(bar=BarId.INPUT)
tau = dynamics.solve_required_effort(torque_A, theta_input_ddot=0.0)
print(f"Par en A para sostener          : {tau:12.1f} mN·mm")

# b) Fuerza perpendicular al acoplador en su extremo (empujar la puerta)
push = Actuator(bar=BarId.COUPLER, point=Point(x=70.0, y=0.0),
                direction=(0.0, 1.0), local_direction=True)
force = dynamics.solve_required_effort(push, theta_input_ddot=0.0)
print(f"Fuerza ⟂ acoplador para sostener: {force:12.1f} mN")

# c) Misma fuerza, pero para abrir acelerando a 50 rad/s²
force_acc = dynamics.solve_required_effort(push, theta_input_ddot=50.0)
print(f"Fuerza para acelerar a 50 rad/s²: {force_acc:12.1f} mN")


# =====================================================================
# 6. REACCIONES EN LAS ARTICULACIONES
# =====================================================================
#
# Newton-Euler por barra con el estado cinemático actual
# (θ, ω, α deben estar resueltos). Convenio:
#   R_A: barra fija -> entrada      R_B: entrada -> acoplador
#   R_C: acoplador -> salida        R_D: barra fija -> salida

title("6. REACCIONES (sosteniendo con la fuerza ⟂ acoplador)")

dynamics.solve_required_effort(push, theta_input_ddot=0.0)
result = FourBarReactions(dynamics).solve(push)
print(f"λ (Newton-Euler) = {result.actuator_value:.1f} mN  "
      f"(comprobación, residuo {result.residual:.1e})")
for joint, r in result.reactions.items():
    print(f"  R_{joint} = ({r[0]:10.1f}, {r[1]:10.1f}) mN   "
          f"|R| = {np.linalg.norm(r):10.1f} mN")


# =====================================================================
# 7. BARRIDO ESTÁTICO: CURVA DE ESFUERZO
# =====================================================================

title("7. CURVA DE ESFUERZO (fuerza ⟂ acoplador)")

print(f"{'θ entrada':>10s} {'F [mN]':>10s} {'|R_A|':>10s} {'|R_B|':>10s} "
      f"{'|R_C|':>10s} {'|R_D|':>10s}")
reactions_solver = FourBarReactions(dynamics)
for angle in np.deg2rad(np.arange(40.0, 151.0, 10.0)):
    kinematics.solve_configuration(angle, BRANCH)
    if kinematics.get_closure_error() > 1e-6:
        print(f"{deg(angle)}   no montable")
        continue
    kinematics.solve_theta_dot(0.0)
    try:
        f = dynamics.solve_required_effort(push, theta_input_ddot=0.0)
    except RuntimeError:
        print(f"{deg(angle)}   singular (punto muerto)")
        continue
    r = reactions_solver.solve(push).reactions
    print(f"{deg(angle)} {f:10.1f} " + " ".join(
        f"{np.linalg.norm(r[j]):10.1f}" for j in "ABCD"))


# =====================================================================
# 8. SIMULACIÓN DEL RETORNO LIBRE
# =====================================================================
#
# Se suelta en reposo en THETA_START; muelles + gravedad + cargas
# externas lo llevan hasta THETA_HOME (si pueden).

title("8. RETORNO LIBRE")

THETA_START = np.deg2rad(150.0)
THETA_HOME = np.deg2rad(40.0)

simulation = ReturnSimulation(dynamics, branch=BRANCH)
trajectory = simulation.run(THETA_START, THETA_HOME, dt=1e-4, t_max=1.0)

print(f"Resultado          : {trajectory.stop_reason}")
print(f"Tiempo             : {trajectory.duration * 1e3:.2f} ms")
print(f"Velocidad en home  : {trajectory.home_velocity:.2f} rad/s")
print(f"E. cinética en home: {trajectory.kinetic_energy[-1]:.1f} µJ")
drift = np.ptp(trajectory.total_energy)
relative = drift / np.max(np.abs(trajectory.kinetic_energy))
print(f"Deriva de energía  : {drift:.2e} µJ ({relative:.1e} relativo; "
      "si es grande, reducir dt)")

# La trayectoria guarda todo para posprocesar o animar:
#   trajectory.time, trajectory.theta_input, trajectory.theta_input_dot,
#   trajectory.theta_input_ddot, trajectory.theta[BarId.COUPLER], ...


# =====================================================================
# 9. INFORME PDF A4
# =====================================================================

title("9. INFORME")

plotter = FourBarPlotter(
    dynamics,
    branch=BRANCH,
    actuator=push,                       # fuerza de la curva de esfuerzo
    units=Units(length="mm", force="mN", energy="µJ", torque="mN·mm"),
)
path = plotter.build_report(
    "output/ejemplo_completo.pdf",
    trajectory,
    # Rango del barrido de esfuerzo (por defecto: de home a inicio)
    theta_range=np.deg2rad(np.linspace(40.0, 150.0, 200)),
    title="Bisagra de puerta · ejemplo completo",
)
print(f"PDF generado: {path}")

# Los gráficos también se pueden usar sueltos en una figura propia:
#   import matplotlib.pyplot as plt
#   fig, ax = plt.subplots()
#   plotter.plot_geometry(ax, np.deg2rad(90))
#   fig.savefig("output/geometria.png", dpi=200)
