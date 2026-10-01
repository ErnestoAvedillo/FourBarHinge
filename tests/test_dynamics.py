import numpy as np
import pytest

from fourbarhinge import (
    Actuator, BarId, FourBarReactions, Point, ReturnSimulation,
)


def set_state(dynamics, theta_input, theta_input_dot=0.0, branch=1):
    kinematics = dynamics.kinematics
    kinematics.solve_configuration(theta_input, branch)
    kinematics.solve_theta_dot(theta_input_dot)


@pytest.mark.parametrize("branch", [1, -1])
def test_loop_closure(dynamics, branch):
    kinematics = dynamics.kinematics
    for deg in range(0, 360, 15):
        kinematics.solve_configuration(np.deg2rad(deg), branch)
        assert kinematics.get_closure_error() < 1e-9


def test_reduced_mass_matches_kinetic_energy(dynamics):
    set_state(dynamics, np.deg2rad(60), 1.7)
    assert np.isclose(0.5 * dynamics.get_reduced_mass() * 1.7**2,
                      dynamics.get_kinetic_energy_direct())


def test_reduced_terms_match_finite_differences(dynamics):
    h = 1e-6

    def at(theta, fn, w=1.3):
        set_state(dynamics, theta, w)
        return fn()

    for deg in (60, 10, -45, 150):
        ti = np.deg2rad(deg)
        d_gravity = (at(ti + h, dynamics.get_gravity_potential_energy)
                     - at(ti - h, dynamics.get_gravity_potential_energy))
        d_springs = (at(ti + h, dynamics.get_spring_potential_energy)
                     - at(ti - h, dynamics.get_spring_potential_energy))
        d_mass = (at(ti + h, dynamics.get_reduced_mass)
                  - at(ti - h, dynamics.get_reduced_mass))
        set_state(dynamics, ti, 1.3)
        assert np.isclose(dynamics.get_reduced_gravity(), d_gravity / (2 * h),
                          rtol=1e-5)
        assert np.isclose(dynamics.get_reduced_spring_force(),
                          -d_springs / (2 * h), rtol=1e-5)
        assert np.isclose(dynamics.get_reduced_coriolis(),
                          0.5 * d_mass / (2 * h) * 1.3**2, rtol=1e-5)


def test_energy_conservation(dynamics):
    trajectory = ReturnSimulation(dynamics, branch=1).run(
        np.deg2rad(60), np.deg2rad(-300), dt=2e-4, t_max=0.04)
    assert len(trajectory.time) > 100
    drift = np.ptp(trajectory.total_energy)
    assert drift < 1e-6 * np.max(np.abs(trajectory.total_energy))


@pytest.mark.parametrize("actuator", [
    Actuator(bar=BarId.INPUT),
    Actuator(bar=BarId.COUPLER, point=Point(x=0.25, y=0.05),
             direction=(0.3, -1.0)),
    Actuator(bar=BarId.COUPLER, point=Point(x=0.45, y=0.0),
             direction=(0.0, 1.0), local_direction=True),
])
def test_newton_euler_matches_reduced_dynamics(dynamics, actuator):
    set_state(dynamics, np.deg2rad(75), 2.0)
    effort = dynamics.solve_required_effort(actuator, theta_input_ddot=4.0)
    result = FourBarReactions(dynamics).solve(actuator)
    assert np.isclose(effort, result.actuator_value)
    assert result.residual < 1e-9


def test_free_motion_reactions_are_consistent(dynamics):
    set_state(dynamics, np.deg2rad(75), 2.0)
    dynamics.solve_theta_input_ddot()
    assert FourBarReactions(dynamics).solve().residual < 1e-9
