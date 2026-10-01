import numpy as np

from fourbarhinge import (
    Actuator, BarId, CompressionSpringDesign, CouplerPathSynthesis,
    EffortTravelCurve, LinkBounds, Point, SpringSynthesis,
    TorsionSpringDesign,
)
from fourbarhinge.synthesis import coupler_point_positions

PARAMS = {"ax": 0.0, "ay": 0.0, "theta_ground": 0.1, "lg": 60.0,
          "li": 25.0, "lc": 70.0, "lo": 55.0, "u": 40.0, "v": 15.0}


def test_vectorised_coupler_matches_kinematics(dynamics):
    kinematics = dynamics.kinematics
    g = kinematics.geometry
    params = {"ax": g.position_ground.x, "ay": g.position_ground.y,
              "theta_ground": kinematics.configuration.theta[BarId.GROUND],
              "lg": g.bar[BarId.GROUND].length,
              "li": g.bar[BarId.INPUT].length,
              "lc": g.bar[BarId.COUPLER].length,
              "lo": g.bar[BarId.OUTPUT].length, "u": 0.1, "v": 0.07}
    theta = np.deg2rad(np.arange(0, 360, 30))
    for branch in (1, -1):
        points, _, _, _ = coupler_point_positions(params, theta, branch)
        for ti, p in zip(theta, points):
            kinematics.solve_configuration(ti, branch)
            expected = kinematics.get_local_to_global(
                BarId.COUPLER, Point(x=0.1, y=0.07))
            assert np.allclose(p, expected)


def test_path_synthesis_recovers_known_path():
    theta = np.deg2rad(np.linspace(150, 40, 7))
    targets, _, _, _ = coupler_point_positions(PARAMS, theta, -1)
    lg = PARAMS["lg"]
    d = Point(x=lg * np.cos(0.1), y=lg * np.sin(0.1))
    synthesis = CouplerPathSynthesis(
        targets, LinkBounds(min_length=10.0, max_length=120.0,
                            min_transmission_angle=0.0),
        ground_pivot_a=Point(x=0.0, y=0.0), ground_pivot_d=d)
    result = synthesis.solve(n_starts=40, seed=1)
    assert result.feasible
    assert result.max_error < 1e-3


def test_effort_curve_matches_dynamics(dynamics):
    curve = EffortTravelCurve(dynamics, Point(x=0.45, y=0.0),
                              np.deg2rad(40), np.deg2rad(120), samples=25)
    effort = curve.effort()
    kinematics = dynamics.kinematics
    for i in range(0, 25, 6):
        kinematics.solve_configuration(curve.theta_input[i], 1)
        kinematics.solve_theta_dot(0.0)
        # Fuerza tangente a la trayectoria de P en el sentido del recorrido
        v = kinematics.get_point_jacobian(
            BarId.COUPLER, Point(x=0.45, y=0.0)) @ \
            kinematics.get_velocity_ratios()
        actuator = Actuator(bar=BarId.COUPLER, point=Point(x=0.45, y=0.0),
                            direction=tuple(v))
        expected = dynamics.solve_required_effort(actuator)
        assert np.isclose(effort[i], expected, rtol=1e-8)


def test_spring_synthesis_recovers_known_springs(dynamics):
    curve = EffortTravelCurve(dynamics, Point(x=0.45, y=0.0),
                              np.deg2rad(40), np.deg2rad(120), samples=60)
    designs_c = [CompressionSpringDesign(
        bar_a=BarId.GROUND, bar_b=BarId.COUPLER, a_x=(0.1, 0.1),
        a_y=(0.0, 0.0), b_x=(0.2, 0.2), b_y=(0.0, 0.0),
        constant=(50.0, 500.0), free_length=(0.3, 0.6))]
    designs_t = [TorsionSpringDesign(bar_a=BarId.GROUND, bar_b=BarId.INPUT,
                                     constant=(0.5, 10.0),
                                     free_angle=(-1.0, 1.0))]
    true_c = designs_c[0].build({"a_x": 0.1, "a_y": 0.0, "b_x": 0.2,
                                 "b_y": 0.0, "constant": 200.0,
                                 "free_length": 0.45})
    true_t = designs_t[0].build({"constant": 3.0, "free_angle": 0.5})
    target = curve.effort([true_c], [true_t])
    ok = np.isfinite(target)
    synthesis = SpringSynthesis(curve, curve.travel[ok], target[ok],
                                designs_c, designs_t)
    result = synthesis.solve(n_starts=10, seed=0)
    assert result.max_error < 1e-4 * np.nanmax(np.abs(target))
