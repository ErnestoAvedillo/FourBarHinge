# fourbarhinge

A Python library to **analyse and design four-bar hinges**: kinematics,
dynamics on top of [modern-robotics](https://pypi.org/project/modern-robotics/),
joint reactions, compression/torsion spring effort curves, free-return
simulation, path synthesis of the coupler point and spring placement for a
target effort–travel curve. Results can be exported as A4 PDF reports.

Number of downloads: [![PyPI Downloads](https://img.shields.io/pypi/dm/fourbarhinge)](https://pypi.org/project/fourbarhinge/)

Visit the library on GitHub https://github.com/ErnestoAvedillo/FourBarHinge and clone the repository using:

    git clone git@github.com:ErnestoAvedillo/FourBarHinge.git

## Features

- **Kinematics**: closed-form position (both assembly branches), velocities,
  accelerations and any point of any bar.
- **Dynamics reduced to the single DOF**: the open-chain mass matrix from
  modern-robotics is mapped to four-bar coordinates (`M_fb = Tᵀ·M_MR·T`) and
  reduced with the velocity ratios Γ:

  `M_red·θ̈ + C_red + G_red = Q_springs + Q_external + λ·Q_actuator`

- **Springs**: compression springs between any two bars (push only) and
  torsion springs in the hinges.
- **Required effort**: torque or force (fixed or following the bar) needed to
  hold or move the mechanism.
- **Joint reactions** in A, B, C, D by Newton–Euler, cross-checked against
  the reduced equation.
- **Free-return simulation** (RK4) from a released position to *home*, with
  energy balance.
- **Path synthesis**: find the link lengths, pivots and coupler point so that
  the coupler point follows a desired travel.
- **Spring synthesis**: place and size compression/torsion springs to match a
  target effort–travel curve.
- **A4 PDF reports** (matplotlib).

## Structure

```
src/fourbarhinge/
├── models/        Data models (pydantic): bars, geometry, springs, loads
├── kinematics/    FourBarKinematics
├── dynamics/      FourBarDynamics (modern-robotics), FourBarReactions
├── simulation/    ReturnSimulation (free return to home)
├── synthesis/     CouplerPathSynthesis, EffortTravelCurve, SpringSynthesis
└── plots/         FourBarPlotter, SynthesisPlotter (A4 PDF reports)

examples/          Complete worked examples
tests/             Tests (pytest)
```

## Installation

I recommend using uv to install the library (https://docs.astral.sh/uv/):

```bash
uv init
uv add fourbarhinge
```

or with pip:

```bash
pip install fourbarhinge
```

## Conventions

```
     B ●────────────● C
      /   coupler    \
   input            output
    /                  \
 A ●────────────────────● D
          ground
```

- Each bar has a local frame: origin at its first joint (A, B or D), x axis
  towards the other joint. Points on a bar (centre of mass, spring anchors,
  loads, coupler point P) are given in that frame.
- Angles are absolute with respect to the ground bar (A → D), in radians,
  counter-clockwise. The output bar is measured from D towards C.
- `branch=±1` selects one of the two assembly solutions.
- **Units are never converted**: use a consistent set. With mm–kg–s, forces
  are mN, torques mN·mm, energies µJ and gravity is `(0, -9810)`.

## Usage

### Analysis

```python
import numpy as np
from fourbarhinge import (
    Bar, Point, FourBarGeometry, BarId, FourBarKinematics, FourBarDynamics,
    CompressionSpring, TorsionSpring, Attachment, Position, Actuator,
    FourBarReactions,
)

def bar(length, mass):
    return Bar(length=length, mass=mass, inertia=mass * length**2 / 12,
               center_of_mass=Point(x=length / 2, y=0.0))

geometry = FourBarGeometry(
    position_ground=Point(x=0.0, y=0.0),
    bar={BarId.GROUND: bar(60, 0.05), BarId.INPUT: bar(25, 0.02),
         BarId.COUPLER: bar(70, 0.06), BarId.OUTPUT: bar(55, 0.04)},
)
kinematics = FourBarKinematics(geometry, theta_ground=0.0)
dynamics = FourBarDynamics(
    kinematics,
    compression_springs=[CompressionSpring(
        free_length=60.0, constant=500.0,
        a=Attachment(bar=BarId.GROUND, position=Position(length=0, angle=0)),
        b=Attachment(bar=BarId.COUPLER, position=Position(length=50, angle=0)),
    )],
    torsion_springs=[TorsionSpring(free_angle=np.deg2rad(20), constant=5e4,
                                   bar_a=BarId.GROUND, bar_b=BarId.INPUT)],
    gravity=(0.0, -9810.0),
)

kinematics.solve_configuration(np.deg2rad(100), branch=-1)
kinematics.solve_theta_dot(0.0)

# Force perpendicular to the coupler, at its end, to hold the position
push = Actuator(bar=BarId.COUPLER, point=Point(x=70, y=0),
                direction=(0, 1), local_direction=True)
force = dynamics.solve_required_effort(push, theta_input_ddot=0.0)

reactions = FourBarReactions(dynamics).solve(push).reactions
print(force, reactions["A"], reactions["B"], reactions["C"], reactions["D"])
```

### Free return to home and PDF report

```python
from fourbarhinge import ReturnSimulation, FourBarPlotter

trajectory = ReturnSimulation(dynamics, branch=-1).run(
    np.deg2rad(150), np.deg2rad(40), dt=1e-4)
print(trajectory.stop_reason, trajectory.home_velocity)

FourBarPlotter(dynamics, branch=-1, actuator=push).build_report(
    "report.pdf", trajectory)
```

The report contains: summary tables, mechanism geometry, effort curve and
joint reactions versus the input angle, angle/velocity/acceleration during
the return and the energy balance.

### Path synthesis

```python
from fourbarhinge import CouplerPathSynthesis, LinkBounds

targets = np.array([[90, 10], [92, 25], [88, 40], [78, 54],
                    [62, 64], [42, 70], [22, 70]], float)

result = CouplerPathSynthesis(
    targets,
    LinkBounds(min_length=15, max_length=120,
               min_transmission_angle=np.deg2rad(30)),
    ground_pivot_a=Point(x=0, y=0),      # optional: fixed pivots
    ground_pivot_d=Point(x=40, y=-10),
).solve(n_starts=60)
print(result.summary())

geometry = result.to_geometry(linear_density=1.6e-4, coupler_mass=0.4)
kinematics = result.to_kinematics(geometry)
```

At least five target points are needed for a well-determined problem; with
fewer points one of the infinite solutions is returned. Fixing the ground
pivots or the input angles (`theta_inputs`) reduces the unknowns.

### Effort–travel curve and spring synthesis

```python
from fourbarhinge import (
    EffortTravelCurve, SpringSynthesis, CompressionSpringDesign,
    TorsionSpringDesign,
)

dynamics = FourBarDynamics(kinematics, gravity=(0, -9810))
curve = EffortTravelCurve(dynamics, result.coupler_point,
                          result.theta_start, result.theta_end,
                          branch=result.branch)

# Effort with any set of springs (fast, vectorised)
effort = curve.effort(compression_springs=[...], torsion_springs=[...])

# Or let the optimiser place and size them: each parameter is a (low, high)
# range, low == high fixes it.
springs = SpringSynthesis(
    curve,
    target_travel=[0, curve.travel[-1]],
    target_effort=[3000, -2000],
    compression_designs=[CompressionSpringDesign(
        bar_a=BarId.GROUND, bar_b=BarId.OUTPUT,
        a_x=(0, 40), a_y=(-15, 15), b_x=(10, 100), b_y=(-5, 5),
        constant=(50, 2000), free_length=(20, 120))],
    torsion_designs=[TorsionSpringDesign(
        bar_a=BarId.GROUND, bar_b=BarId.INPUT,
        constant=(0, 2e5), free_angle=(-np.pi, np.pi))],
    min_spring_length=15,
).solve()
print(springs.summary())
```

The effort is the static force at the coupler point, tangent to its path:
`F > 0` means it must be pushed along the travel, `F < 0` means the hinge
moves by itself. The spring working ranges (length, force, deflection) are
reported so the springs can then be designed, for example with
[springcalc](https://pypi.org/project/springcalc/).

## Examples

```bash
uv run python examples/ejemplo_completo.py    # every analysis step
uv run python examples/informe_bisagra.py     # dynamic analysis PDF report
uv run python examples/sintesis_bisagra.py    # path + spring synthesis
```

PDF reports are written to `output/`.

## Development

```bash
uv sync
make unittest        # run the tests
make build           # bump patch version and build sdist + wheel
make test            # build and upload to TestPyPI
make pypi            # build and upload to PyPI
```

Uploads use the `pypi` / `testpypi` repositories of `~/.pypirc`.

## Modelling notes

- Compression springs only push: beyond their free length their force is 0.
- Torsion spring deflection is wrapped to (-π, π].
- `inertia` is the moment of inertia about the centre of mass.
- Dead points (singular Jacobian) raise an error; non-assemblable input
  angles are detected with `FourBarKinematics.get_closure_error()`.
- No friction is modelled.

## License

MIT
