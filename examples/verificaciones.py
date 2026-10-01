
from fourbarhinge.models.barra import Bar, Point, FourBarGeometry
from fourbarhinge.dynamics.dynamics import FourBarDynamics
from fourbarhinge.kinematics.kinematics import FourBarKinematics
import modern_robotics as mr
import numpy as np
from fourbarhinge.models.constants import BarId
from fourbarhinge.models.position import Position, Attachment
from fourbarhinge.models.compression_spring import CompressionSpring
from fourbarhinge.models.torsion_spring import TorsionSpring
from fourbarhinge.models.external_force import Actuator
from fourbarhinge.dynamics.reactions import FourBarReactions

origin = Point(x=10, y=5)
barra = Bar(length=40.0, mass=1.0, inertia=100.0,
            center_of_mass=Point(
                                 x=20.0,
                                 y=0.0,
            ))
geometry = FourBarGeometry(
    position_ground=origin,
    bar={BarId.GROUND: barra,
         BarId.INPUT: barra,
         BarId.COUPLER: barra,
         BarId.OUTPUT: barra
         })

mechanism = FourBarKinematics(geometry=geometry, theta_ground=np.deg2rad(30))
mechanism.solve_configuration(theta_input=3.14159 / 2, branch=-1)
error = mechanism.configuration.theta[BarId.OUTPUT] - 3.14159 / 2
print(f"El ángulo de salida es:{mechanism.configuration.theta[BarId.OUTPUT]}")
print(f"El error de salida es:{error}")
assert mechanism.configuration.theta[BarId.OUTPUT] > 0.001, "incorrect resut"
mechanism.get_jacobian()
mechanism.solve_theta_dot(0.1)
print(f"The velocities for the geometry is thta out = \
        {mechanism.theta_dot.theta_dot[BarId.OUTPUT]}")
print(f"The velocities for the geometry is thta coup = \
        {mechanism.theta_dot.theta_dot[BarId.COUPLER]}")
point_local = Point(x=10.0, y=5.0)

point_global = mechanism.get_local_to_global(
    BarId.COUPLER,
    point_local
)

point_local_recovered = mechanism.get_global_to_local(
    BarId.COUPLER,
    Point(
        x=point_global[0],
        y=point_global[1]
    )
)

assert np.allclose(
    point_local_recovered,
    np.array([point_local.x, point_local.y])
)
for bar in BarId:
    point_local = Point(x=10.0, y=5.0)

    point_global = mechanism.get_local_to_global(
        bar,
        point_local
    )

    recovered = mechanism.get_global_to_local(
        bar,
        Point(
            x=point_global[0],
            y=point_global[1]
        )
    )

    assert np.allclose(
        recovered,
        np.array([10.0, 5.0])
    )

dynamics = FourBarDynamics(mechanism)  # You need to define `four_bar_params` before this line
Mlist = dynamics.get_mlist()
Slist = dynamics.get_slist()
thetalist = dynamics.get_mr_thetalist()

M_home = dynamics.get_mr_home()

T_end = mr.FKinSpace(
    M_home,
    Slist,
    thetalist,
)

print("M home:")
print(M_home)

print("thetalist:")
print(thetalist)

print("T end:")
print(T_end)


def test_mr_kinematics(dynamics: FourBarDynamics) -> None:

    kinematics = dynamics.kinematics
    geometry = dynamics.geometry

    Li = geometry.bar[BarId.INPUT].length
    Lc = geometry.bar[BarId.COUPLER].length
    Lg = geometry.bar[BarId.GROUND].length

    Slist = dynamics.get_slist()
    thetalist = dynamics.get_mr_thetalist()

    ti = kinematics.configuration.theta[BarId.INPUT]
    tc = kinematics.configuration.theta[BarId.COUPLER]

    # ---------------------------
    # B
    # ---------------------------

    M_B = np.eye(4)
    M_B[0, 3] = Li

    T_B = mr.FKinSpace(
        M_B,
        Slist[:, :1],
        thetalist[:1],
    )

    B_mr = T_B[:2, 3]

    B_kin = np.array([
        Li * np.cos(ti),
        Li * np.sin(ti),
    ])

    print("\nB:")
    print("  MR :", B_mr)
    print("  KIN:", B_kin)

    assert np.allclose(B_mr, B_kin, atol=1e-8)

    # ---------------------------
    # C
    # ---------------------------

    M_C = np.eye(4)
    M_C[0, 3] = Li + Lc

    T_C = mr.FKinSpace(
        M_C,
        Slist[:, :2],
        thetalist[:2],
    )

    C_mr = T_C[:2, 3]

    C_kin = np.array([
        Li * np.cos(ti) + Lc * np.cos(tc),
        Li * np.sin(ti) + Lc * np.sin(tc),
    ])

    print("\nC:")
    print("  MR :", C_mr)
    print("  KIN:", C_kin)

    assert np.allclose(C_mr, C_kin, atol=1e-8)

    # ---------------------------
    # D
    # ---------------------------

    M_D = dynamics.get_mr_home()

    T_D = mr.FKinSpace(
        M_D,
        Slist,
        thetalist,
    )

    D_mr = T_D[:2, 3]

    D_kin = np.array([
        Lg,
        0.0,
    ])

    print("\nD:")
    print("  MR :", D_mr)
    print("  KIN:", D_kin)

    assert np.allclose(D_mr, D_kin, atol=1e-8)

    print("\nMR kinematics OK")

dynamics = FourBarDynamics(mechanism)

test_mr_kinematics(dynamics)

M = dynamics.get_mass_matrix()
print("Mass matrix:")
print(M)
assert M.shape == (3, 3)
assert np.allclose(M, M.T, atol=1e-10)

print(dynamics.get_mr_dthetalist())
print(dynamics.get_mr_ddthetalist())

print(dynamics.get_kinetic_energy())

v_A = mechanism.get_point_velocity(
                                   BarId.INPUT,
                                   Point(x=0.0, y=0.0),
                                   )

v_D = mechanism.get_point_velocity(
                                   BarId.OUTPUT,
                                   Point(x=0.0, y=0.0),
                                   )

assert np.allclose(v_A, np.zeros(2))
assert np.allclose(v_D, np.zeros(2))

Li = geometry.bar[BarId.INPUT].length
Lc = geometry.bar[BarId.COUPLER].length
Lo = geometry.bar[BarId.OUTPUT].length

# A
v_A = mechanism.get_point_velocity(
    BarId.INPUT,
    Point(x=0.0, y=0.0),
)

assert np.allclose(v_A, np.zeros(2))


# B calculated from INPUT
v_B_input = mechanism.get_point_velocity(
    BarId.INPUT,
    Point(x=Li, y=0.0),
)

# B calculated from COUPLER
v_B_coupler = mechanism.get_point_velocity(
    BarId.COUPLER,
    Point(x=0.0, y=0.0),
)

assert np.allclose(
    v_B_input,
    v_B_coupler,
    atol=1e-9,
)


# C calculated from COUPLER
v_C_coupler = mechanism.get_point_velocity(
    BarId.COUPLER,
    Point(x=Lc, y=0.0),
)

# C calculated from OUTPUT
v_C_output = mechanism.get_point_velocity(
    BarId.OUTPUT,
    Point(x=Lo, y=0.0),
)

assert np.allclose(
    v_C_coupler,
    v_C_output,
    atol=1e-9,
)

T_mr = dynamics.get_kinetic_energy_mr()

T_direct = dynamics.get_kinetic_energy_direct()

print(f"Kinetic energy MR:     {T_mr}")
print(f"Kinetic energy direct: {T_direct}")
print(f"Error:                 {T_mr - T_direct}")

assert np.isclose(
    T_mr,
    T_direct,
    rtol=1e-9,
    atol=1e-9,
)

# ---------------------------------------------------------------
# Dinámica reducida + muelles + reacciones
# (unidades coherentes: aquí mm, kg, s -> g en mm/s², fuerzas en
#  kg·mm/s² = mN, pares en mN·mm)
# ---------------------------------------------------------------

spring_dynamics = FourBarDynamics(
    mechanism,
    compression_springs=[
        CompressionSpring(
            free_length=50.0,
            constant=500.0,
            a=Attachment(bar=BarId.GROUND,
                         position=Position(length=10.0, angle=0.0)),
            b=Attachment(bar=BarId.COUPLER,
                         position=Position(length=20.0, angle=0.0)),
        ),
    ],
    torsion_springs=[
        TorsionSpring(free_angle=np.deg2rad(60), constant=2000.0,
                      bar_a=BarId.GROUND, bar_b=BarId.INPUT),
    ],
    gravity=(0.0, -9810.0),
)

mechanism.solve_theta_dot(0.0)
print("M_red:", spring_dynamics.get_reduced_mass())
print("C_red:", spring_dynamics.get_reduced_coriolis())
print("G_red:", spring_dynamics.get_reduced_gravity())
print("Q_springs:", spring_dynamics.get_reduced_spring_force())

theta_input_ddot = spring_dynamics.solve_theta_input_ddot()
print(f"Sin actuador -> ẗi = {theta_input_ddot}")

# Fuerza vertical en el extremo del acoplador para mantener el equilibrio
hand = Actuator(bar=BarId.COUPLER, point=Point(x=40.0, y=0.0),
                direction=(0.0, -1.0))
force = spring_dynamics.solve_required_effort(hand, theta_input_ddot=0.0)
result = FourBarReactions(spring_dynamics).solve(hand)
print(f"Fuerza necesaria: {force} (Newton-Euler: {result.actuator_value})")
for joint, reaction in result.reactions.items():
    print(f"  Reacción {joint}: {reaction}  |R| = {np.linalg.norm(reaction)}")
assert np.isclose(force, result.actuator_value)
assert result.residual < 1e-6
