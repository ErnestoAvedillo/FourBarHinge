"""
fourbarhinge: cinemática, dinámica y síntesis de bisagras de 4 barras.
"""

from importlib.metadata import PackageNotFoundError, version

from .models.constants import BarId
from .models.barra import Bar, Point, FourBarGeometry
from .models.position import Position, Attachment
from .models.compression_spring import CompressionSpring
from .models.torsion_spring import TorsionSpring
from .models.external_force import ExternalForce, Actuator
from .kinematics.kinematics import FourBarKinematics
from .dynamics.dynamics import FourBarDynamics
from .dynamics.reactions import FourBarReactions, ReactionResult
from .simulation.return_simulation import ReturnSimulation, ReturnTrajectory
from .plots.plots import FourBarPlotter, Units
from .plots.synthesis_plots import SynthesisPlotter
from .synthesis import (
    CouplerPathSynthesis, LinkBounds, PathSynthesisResult,
    CouplerMotionSynthesis, MotionSynthesisResult, EffortTravelCurve, SpringSynthesis, SpringSynthesisResult,
    CompressionSpringDesign, TorsionSpringDesign,
)

try:
    __version__ = version("fourbarhinge")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "BarId", "Bar", "Point", "FourBarGeometry", "Position", "Attachment",
    "CompressionSpring", "TorsionSpring", "ExternalForce", "Actuator",
    "FourBarKinematics", "FourBarDynamics", "FourBarReactions",
    "ReactionResult", "ReturnSimulation", "ReturnTrajectory",
    "FourBarPlotter", "Units", "SynthesisPlotter", "CouplerPathSynthesis", "LinkBounds",
    "PathSynthesisResult", "CouplerMotionSynthesis", "MotionSynthesisResult",
    "EffortTravelCurve", "SpringSynthesis",
    "SpringSynthesisResult", "CompressionSpringDesign",
    "TorsionSpringDesign",
]
