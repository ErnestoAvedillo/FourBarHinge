
"""
Gráficos del cuadrilátero articulado e informe PDF en A4.

  plotter = FourBarPlotter(dynamics, branch=1, actuator=actuator)
  trajectory = ReturnSimulation(dynamics, branch=1).run(start, home)
  plotter.build_report("output/informe.pdf", trajectory)
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from ..dynamics.dynamics import FourBarDynamics  # noqa: E402
from ..dynamics.reactions import FourBarReactions  # noqa: E402
from ..models.barra import Point  # noqa: E402
from ..models.constants import BarId  # noqa: E402
from ..models.external_force import Actuator  # noqa: E402
from ..simulation.return_simulation import ReturnTrajectory  # noqa: E402

A4_PORTRAIT = (8.27, 11.69)

# Paleta (dataviz, modo claro): identidad fija por entidad.
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e5e4e0"
SURFACE = "#ffffff"
BAR_COLOR = {
    BarId.GROUND: INK_2,
    BarId.INPUT: "#2a78d6",
    BarId.COUPLER: "#eb6834",
    BarId.OUTPUT: "#1baf7a",
}
SPRING_COLOR = "#4a3aa7"
JOINT_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
ENERGY_COLOR = {
    "kinetic": "#2a78d6",
    "springs": "#eb6834",
    "gravity": "#1baf7a",
    "external": "#4a3aa7",
    "total": INK,
}
BAR_NAME = {
    BarId.GROUND: "Fija",
    BarId.INPUT: "Entrada",
    BarId.COUPLER: "Acoplador",
    BarId.OUTPUT: "Salida",
}
HINGE_OF = {
    frozenset((BarId.GROUND, BarId.INPUT)): "A",
    frozenset((BarId.INPUT, BarId.COUPLER)): "B",
    frozenset((BarId.COUPLER, BarId.OUTPUT)): "C",
    frozenset((BarId.OUTPUT, BarId.GROUND)): "D",
}


@dataclass
class Units:
    """Etiquetas de unidades (los cálculos no convierten nada)."""
    length: str = "mm"
    force: str = "mN"
    energy: str = "µJ"
    torque: str = "mN·mm"


def style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.6)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def new_page(title: str, subtitle: str = "") -> Figure:
    fig = plt.figure(figsize=A4_PORTRAIT, facecolor=SURFACE)
    fig.text(0.08, 0.955, title, fontsize=17, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.08, 0.93, subtitle, fontsize=9.5, color=INK_2)
    return fig


def footer(fig: Figure, page: int) -> None:
    fig.text(0.08, 0.03, "Bisagra de 4 barras · análisis dinámico",
             fontsize=7.5, color=INK_3)
    fig.text(0.92, 0.03, str(page), fontsize=7.5, color=INK_3, ha="right")


class FourBarPlotter:
    def __init__(
        self,
        dynamics: FourBarDynamics,
        branch: int = 1,
        actuator: Actuator | None = None,
        units: Units | None = None,
        coupler_point: Point | None = None,
    ):
        """
        coupler_point: punto P del acoplador (local) fuera de la línea BC;
        si se da, el acoplador se dibuja como el triángulo rígido B-C-P.
        """
        self.dynamics = dynamics
        self.kinematics = dynamics.kinematics
        self.geometry = dynamics.geometry
        self.branch = branch
        self.actuator = actuator
        self.units = units or Units()
        self.coupler_point = coupler_point

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    def set_static_state(self, theta_input: float) -> bool:
        """Configura el mecanismo en reposo. False si no es montable."""
        self.kinematics.solve_configuration(theta_input, self.branch)
        if self.kinematics.get_closure_error() > 1e-6:
            return False
        self.kinematics.solve_theta_dot(0.0)
        return True

    # ------------------------------------------------------------------
    # Geometría
    # ------------------------------------------------------------------

    def plot_geometry(
        self,
        ax: Axes,
        theta_input: float,
        ghost_theta_input: float | None = None,
        show_actuator: bool = True,
    ) -> None:
        """Dibuja el mecanismo en theta_input (y en gris otra posición)."""
        style_axes(ax)
        ax.set_aspect("equal")
        if ghost_theta_input is not None and self.set_static_state(
                ghost_theta_input):
            self.draw_mechanism(ax, ghost=True)
        if not self.set_static_state(theta_input):
            raise RuntimeError("theta_input outside assembly range")
        self.draw_springs(ax)
        self.draw_mechanism(ax, ghost=False)
        if show_actuator and self.actuator is not None:
            self.draw_actuator(ax)
        ax.set_xlabel(f"x [{self.units.length}]")
        ax.set_ylabel(f"y [{self.units.length}]")
        ax.margins(0.12)

    def draw_mechanism(self, ax: Axes, ghost: bool) -> None:
        joints = self.kinematics.get_joint_positions()
        segments = {
            BarId.GROUND: ("A", "D"),
            BarId.INPUT: ("A", "B"),
            BarId.COUPLER: ("B", "C"),
            BarId.OUTPUT: ("D", "C"),
        }
        for bar_id, (p, q) in segments.items():
            xs = [joints[p][0], joints[q][0]]
            ys = [joints[p][1], joints[q][1]]
            if ghost:
                ax.plot(xs, ys, color=GRID, linewidth=3.0,
                        solid_capstyle="round", zorder=1)
                continue
            ax.plot(xs, ys, color=BAR_COLOR[bar_id], linewidth=3.2,
                    solid_capstyle="round", zorder=3,
                    label=BAR_NAME[bar_id])
            bar = self.geometry.bar[bar_id]
            if bar_id != BarId.GROUND and bar.center_of_mass is not None:
                com = self.kinematics.get_local_to_global(
                    bar_id, bar.center_of_mass)
                ax.plot(*com, marker="o", markersize=5, color=SURFACE,
                        markeredgecolor=BAR_COLOR[bar_id],
                        markeredgewidth=1.5, zorder=5)
        if self.coupler_point is not None:
            p = self.kinematics.get_local_to_global(BarId.COUPLER,
                                                    self.coupler_point)
            triangle = np.array([joints["B"], joints["C"], p])
            ax.fill(triangle[:, 0], triangle[:, 1],
                    color=GRID if ghost else BAR_COLOR[BarId.COUPLER],
                    alpha=0.5 if ghost else 0.15, linewidth=0, zorder=1)
            if not ghost:
                ax.plot(*p, marker="D", markersize=6,
                        color=BAR_COLOR[BarId.COUPLER], zorder=6)
                ax.annotate("P", p, xytext=(7, 7),
                            textcoords="offset points", fontsize=10,
                            fontweight="bold", color=INK, zorder=7)
        if ghost:
            return
        for name, p in joints.items():
            ax.plot(*p, marker="o", markersize=9, color=SURFACE,
                    markeredgecolor=INK, markeredgewidth=1.4, zorder=6)
            ax.annotate(name, p, xytext=(7, 7), textcoords="offset points",
                        fontsize=10, fontweight="bold", color=INK, zorder=7)
        # Pivotes fijos
        scale = self.geometry.bar[BarId.GROUND].length * 0.05
        for name in ("A", "D"):
            p = joints[name]
            ax.plot([p[0] - scale, p[0] + scale], [p[1] - scale] * 2,
                    color=INK_2, linewidth=1.2, zorder=2)
            for k in np.linspace(-scale, scale, 5):
                ax.plot([p[0] + k, p[0] + k - 0.5 * scale],
                        [p[1] - scale, p[1] - 1.6 * scale],
                        color=INK_3, linewidth=0.7, zorder=2)

    def draw_springs(self, ax: Axes) -> None:
        for spring in self.dynamics.compression_springs:
            p_a = self.kinematics.get_local_to_global(
                BarId(spring.a.bar), Point(x=spring.a.position.to_array()[0],
                                           y=spring.a.position.to_array()[1]))
            p_b = self.kinematics.get_local_to_global(
                BarId(spring.b.bar), Point(x=spring.b.position.to_array()[0],
                                           y=spring.b.position.to_array()[1]))
            xs, ys = zigzag(p_a, p_b, turns=8,
                            width=0.06 * np.linalg.norm(p_b - p_a))
            ax.plot(xs, ys, color=SPRING_COLOR, linewidth=1.2, zorder=2,
                    label="Muelle compresión")
        joints = self.kinematics.get_joint_positions()
        size = 0.08 * self.geometry.bar[BarId.GROUND].length
        for spring in self.dynamics.torsion_springs:
            hinge = HINGE_OF.get(
                frozenset((BarId(spring.bar_a), BarId(spring.bar_b))))
            if hinge is None:
                continue
            t = np.linspace(0, 4 * np.pi, 120)
            r = size * (0.25 + 0.75 * t / t[-1])
            p = joints[hinge]
            ax.plot(p[0] + r * np.cos(t), p[1] + r * np.sin(t),
                    color=SPRING_COLOR, linewidth=1.0, zorder=2,
                    label="Muelle torsión")

    def draw_actuator(self, ax: Axes) -> None:
        unit = self.dynamics.get_actuator_unit_load(self.actuator)
        if not np.any(unit.force):
            return
        length = 0.3 * self.geometry.bar[BarId.COUPLER].length
        tail = unit.point - unit.force * length
        ax.annotate("", xy=unit.point, xytext=tail, zorder=8,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6,
                                    mutation_scale=14))
        ax.annotate("F", tail, xytext=(-12, 0), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=INK)
        # Punto invisible para que la flecha entre en los límites
        ax.plot(*tail, alpha=0.0)

    # ------------------------------------------------------------------
    # Curva de esfuerzo
    # ------------------------------------------------------------------

    def compute_effort_curve(
        self, theta_range: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Fuerza estática del actuador y |reacciones| para cada ángulo."""
        if self.actuator is None:
            raise ValueError("An actuator is required for the effort curve")
        reactions_solver = FourBarReactions(self.dynamics)
        force = np.full(len(theta_range), np.nan)
        reactions = {j: np.full(len(theta_range), np.nan) for j in "ABCD"}
        for i, ti in enumerate(theta_range):
            if not self.set_static_state(ti):
                continue
            try:
                force[i] = self.dynamics.solve_required_effort(
                    self.actuator, theta_input_ddot=0.0)
                result = reactions_solver.solve(self.actuator)
            except (RuntimeError, np.linalg.LinAlgError):
                force[i] = np.nan
                continue
            for joint, value in result.reactions.items():
                reactions[joint][i] = np.linalg.norm(value)
        return force, reactions

    def plot_effort_curve(
        self, ax: Axes, theta_range: np.ndarray, force: np.ndarray
    ) -> None:
        style_axes(ax)
        deg = np.rad2deg(theta_range)
        ax.plot(deg, force, color=BAR_COLOR[BarId.COUPLER], linewidth=1.8)
        ax.axhline(0.0, color=INK_3, linewidth=0.8)
        if np.any(np.isfinite(force)):
            i = int(np.nanargmax(np.abs(force)))
            ax.plot(deg[i], force[i], marker="o", markersize=7,
                    color=BAR_COLOR[BarId.COUPLER],
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"{fmt(force[i])} {self.units.force}",
                        (deg[i], force[i]), xytext=(8, 6),
                        textcoords="offset points", fontsize=8.5, color=INK)
        ax.set_xlabel("θ entrada [°]")
        ax.set_ylabel(f"Fuerza F [{self.units.force}]")

    def plot_reactions(
        self, ax: Axes, theta_range: np.ndarray,
        reactions: dict[str, np.ndarray],
    ) -> None:
        style_axes(ax)
        deg = np.rad2deg(theta_range)
        labels = []
        for joint, values in reactions.items():
            ax.plot(deg, values, color=JOINT_COLOR[joint], linewidth=1.6,
                    label=f"|R{joint}|")
            finite = np.where(np.isfinite(values))[0]
            if len(finite):
                labels.append((joint, values[finite[-1]]))
        ax.set_xlabel("θ entrada [°]")
        ax.set_ylabel(f"Reacción [{self.units.force}]")
        ax.set_ylim(bottom=0)
        end_labels(ax, deg[-1], labels)
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left",
                  bbox_to_anchor=(0, 1.0), labelcolor=INK_2)

    # ------------------------------------------------------------------
    # Retorno a home
    # ------------------------------------------------------------------

    def plot_series(
        self, ax: Axes, x: np.ndarray, y: np.ndarray, color: str,
        ylabel: str, value_fmt: str,
    ) -> None:
        style_axes(ax)
        ax.plot(x, y, color=color, linewidth=1.8)
        ax.plot(x[-1], y[-1], marker="o", markersize=7, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(value_fmt.format(y[-1]), (x[-1], y[-1]), xytext=(-8, 8),
                    textcoords="offset points", fontsize=8.5, color=INK,
                    ha="right")
        ax.set_ylabel(ylabel)

    def plot_energy(self, ax: Axes, trajectory: ReturnTrajectory) -> None:
        style_axes(ax)
        t_ms = trajectory.time * 1e3
        series = {
            "kinetic": ("Cinética", trajectory.kinetic_energy),
            "springs": ("Muelles", trajectory.spring_energy),
            "gravity": ("Gravedad (ref. inicial)",
                        trajectory.gravity_energy
                        - trajectory.gravity_energy[0]),
        }
        if np.ptp(trajectory.external_energy) > 0.0:
            series["external"] = ("Cargas externas (ref. inicial)",
                                  trajectory.external_energy
                                  - trajectory.external_energy[0])
        total = sum(values for _, values in series.values())
        series["total"] = ("Total", total)
        for key, (label, values) in series.items():
            ax.plot(t_ms, values, color=ENERGY_COLOR[key],
                    linewidth=2.2 if key == "total" else 1.6, label=label)
        ax.axhline(0.0, color=INK_3, linewidth=0.8)
        ax.set_xlabel("t [ms]")
        ax.set_ylabel(f"Energía [{self.units.energy}]")
        end_labels(ax, t_ms[-1], [(label.split(" ")[0], values[-1])
                                  for label, values in series.values()])
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left",
                  bbox_to_anchor=(0, 1.0), labelcolor=INK_2)

    # ------------------------------------------------------------------
    # Informe A4
    # ------------------------------------------------------------------

    def build_report(
        self,
        path: str | Path,
        trajectory: ReturnTrajectory,
        theta_range: np.ndarray | None = None,
        title: str = "Bisagra de 4 barras",
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if theta_range is None:
            lo = min(trajectory.theta_start, trajectory.theta_home)
            hi = max(trajectory.theta_start, trajectory.theta_home)
            theta_range = np.linspace(lo, hi, 200)

        with PdfPages(path) as pdf:
            pages = [
                self.page_summary(title, trajectory),
                self.page_geometry(trajectory),
                self.page_effort(theta_range),
                self.page_motion(trajectory),
                self.page_energy(trajectory),
            ]
            for number, fig in enumerate(pages, start=1):
                footer(fig, number)
                pdf.savefig(fig)
                plt.close(fig)
            info = pdf.infodict()
            info["Title"] = title
            info["Subject"] = "Análisis dinámico del cuadrilátero articulado"
        return path

    def page_summary(self, title: str, trajectory: ReturnTrajectory) -> Figure:
        fig = new_page(title, f"Informe generado el {date.today():%d/%m/%Y}")
        u = self.units

        def deg(x: float) -> str:
            return f"{np.rad2deg(x):.1f}°"

        tiles = [
            ("Velocidad al llegar a home",
             f"{trajectory.home_velocity:.2f} rad/s"),
            ("Tiempo de retorno", f"{trajectory.duration * 1e3:.1f} ms"),
            ("Energía cinética en home",
             f"{fmt(trajectory.kinetic_energy[-1])} {u.energy}"),
        ]
        for k, (label, value) in enumerate(tiles):
            x = 0.08 + k * 0.29
            fig.text(x, 0.86, label, fontsize=8.5, color=INK_2)
            fig.text(x, 0.83, value, fontsize=15, fontweight="bold",
                     color=INK)
        if not trajectory.reached_home:
            fig.text(0.08, 0.80, f"Atención: no alcanza home "
                     f"({trajectory.stop_reason}).", fontsize=9,
                     color="#e34948")

        rows = [("Barra", f"L [{u.length}]", "m [kg]",
                 f"I_G [kg·{u.length}²]", f"COM [{u.length}]")]
        for bar_id in BarId:
            bar = self.geometry.bar[bar_id]
            com = (f"({bar.center_of_mass.x:g}, {bar.center_of_mass.y:g})"
                   if bar.center_of_mass else "-")
            rows.append((BAR_NAME[bar_id], f"{bar.length:g}",
                         f"{bar.mass:g}" if bar.mass is not None else "-",
                         f"{bar.inertia:g}" if bar.inertia is not None
                         else "-", com))
        y = self.draw_table(fig, 0.74, "Geometría y masas", rows)

        rows = [("Tipo", "Entre", "k", "Libre")]
        for s in self.dynamics.compression_springs:
            rows.append(("Compresión",
                         f"{BAR_NAME[BarId(s.a.bar)]} – "
                         f"{BAR_NAME[BarId(s.b.bar)]}",
                         f"{s.constant:g} {u.force}/{u.length}",
                         f"{s.free_length:g} {u.length}"))
        for s in self.dynamics.torsion_springs:
            rows.append(("Torsión",
                         f"{BAR_NAME[BarId(s.bar_a)]} – "
                         f"{BAR_NAME[BarId(s.bar_b)]}",
                         f"{s.constant:g} {u.torque}/rad",
                         deg(s.free_angle)))
        if len(rows) == 1:
            rows.append(("-", "-", "-", "-"))
        y = self.draw_table(fig, y - 0.04, "Muelles", rows)

        g = self.dynamics.gravity
        rows = [
            ("Parámetro", "Valor"),
            ("θ inicial (suelta en reposo)", deg(trajectory.theta_start)),
            ("θ home", deg(trajectory.theta_home)),
            ("Ángulo de la barra fija",
             deg(self.kinematics.configuration.theta[BarId.GROUND])),
            ("Rama de montaje", f"{self.branch:+d}"),
            ("Gravedad", f"({g[0]:g}, {g[1]:g}) {u.length}/s²"),
            ("Resultado", trajectory.stop_reason),
        ]
        if self.actuator is not None:
            a = self.actuator
            ref = "local" if a.local_direction else "global"
            what = ("par" if a.direction is None else
                    f"fuerza dir. {a.direction} ({ref})")
            rows.append(("Actuador",
                         f"{what} en {BAR_NAME[a.bar]} "
                         f"({a.point.x:g}, {a.point.y:g})"))
        self.draw_table(fig, y - 0.04, "Simulación", rows)
        return fig

    def draw_table(
        self, fig: Figure, top: float, heading: str,
        rows: list[tuple[str, ...]],
    ) -> float:
        fig.text(0.08, top, heading, fontsize=12, fontweight="bold",
                 color=INK)
        row_h = 0.024
        y = top - 0.035
        n = len(rows[0])
        xs = 0.08 + np.arange(n) * (0.84 / n)
        if n == 2:
            xs = [0.08, 0.45]
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                fig.text(xs[c], y, cell, fontsize=8.5,
                         color=INK_2 if r == 0 else INK,
                         fontweight="bold" if r == 0 else "normal")
            if r == 0:
                fig.add_artist(plt.Line2D([0.08, 0.92], [y - 0.007] * 2,
                                          color=GRID, linewidth=0.8))
            y -= row_h
        return y

    def page_geometry(self, trajectory: ReturnTrajectory) -> Figure:
        fig = new_page(
            "Geometría del mecanismo",
            "En gris, la otra posición. ○ centros de masas; "
            "F: fuerza externa sobre el acoplador.")
        panels = [
            ((0.1, 0.53, 0.82, 0.36), trajectory.theta_start,
             trajectory.theta_home, "Posición inicial"),
            ((0.1, 0.1, 0.82, 0.36), trajectory.theta_home,
             trajectory.theta_start, "Home"),
        ]
        axes = []
        for box, theta, ghost, name in panels:
            ax = fig.add_axes(box)
            self.plot_geometry(ax, theta, ghost_theta_input=ghost)
            ax.set_title(f"{name} · θ entrada = {np.rad2deg(theta):.1f}°",
                         fontsize=10.5, loc="left", pad=8)
            axes.append(ax)
        # Mismos límites en ambos paneles para poder comparar
        xs = [lim for ax in axes for lim in ax.get_xlim()]
        ys = [lim for ax in axes for lim in ax.get_ylim()]
        for ax in axes:
            ax.set_xlim(min(xs), max(xs))
            ax.set_ylim(min(ys), max(ys))
            ax.set_adjustable("box")
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        fig.legend(unique.values(), unique.keys(), frameon=False, fontsize=8,
                   ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.05),
                   labelcolor=INK_2)
        return fig

    def page_effort(self, theta_range: np.ndarray) -> Figure:
        fig = new_page(
            "Esfuerzo de la fuerza externa sobre el acoplador",
            "Equilibrio estático (ṫi = 0, ẗi = 0) con muelles y gravedad. "
            "F > 0: en el sentido de la flecha.")
        force, reactions = self.compute_effort_curve(theta_range)
        ax = fig.add_axes((0.12, 0.55, 0.8, 0.33))
        self.plot_effort_curve(ax, theta_range, force)
        ax.set_title("Fuerza necesaria para mantener la posición",
                     fontsize=10.5, loc="left", pad=10)
        ax = fig.add_axes((0.12, 0.1, 0.8, 0.33))
        self.plot_reactions(ax, theta_range, reactions)
        ax.set_title("Reacciones en las articulaciones", fontsize=10.5,
                     loc="left", pad=24)
        return fig

    def page_motion(self, trajectory: ReturnTrajectory) -> Figure:
        fig = new_page(
            "Retorno libre hasta home",
            f"Se suelta en reposo en θ = "
            f"{np.rad2deg(trajectory.theta_start):.1f}° y los muelles lo "
            f"llevan a θ = {np.rad2deg(trajectory.theta_home):.1f}°.")
        t_ms = trajectory.time * 1e3
        boxes = [(0.12, 0.66, 0.8, 0.22), (0.12, 0.38, 0.8, 0.22),
                 (0.12, 0.1, 0.8, 0.22)]
        series = [
            (np.rad2deg(trajectory.theta_input), BAR_COLOR[BarId.INPUT],
             "θ entrada [°]", "{:.1f}°"),
            (trajectory.theta_input_dot, BAR_COLOR[BarId.INPUT],
             "Velocidad ṫi [rad/s]", "{:.2f} rad/s"),
            (trajectory.theta_input_ddot, BAR_COLOR[BarId.INPUT],
             "Aceleración ẗi [rad/s²]", "{:.0f} rad/s²"),
        ]
        for box, (y, color, label, fmt) in zip(boxes, series):
            ax = fig.add_axes(box)
            self.plot_series(ax, t_ms, y, color, label, fmt)
        ax.set_xlabel("t [ms]")
        return fig

    def page_energy(self, trajectory: ReturnTrajectory) -> Figure:
        fig = new_page(
            "Energía durante el retorno",
            "La energía de los muelles y la gravedad se convierte en "
            "energía cinética. Sin rozamiento el total es constante.")
        ax = fig.add_axes((0.12, 0.5, 0.74, 0.36))
        self.plot_energy(ax, trajectory)
        released = (trajectory.spring_energy[0]
                    - trajectory.spring_energy[-1])
        gravity = (trajectory.gravity_energy[0]
                   - trajectory.gravity_energy[-1])
        external = (trajectory.external_energy[0]
                    - trajectory.external_energy[-1])
        drift = np.ptp(trajectory.total_energy)
        rows = [
            ("Concepto", "Valor"),
            ("Energía liberada por los muelles",
             f"{fmt(released, 1)} {self.units.energy}"),
            ("Energía aportada por la gravedad",
             f"{fmt(gravity, 1)} {self.units.energy}"),
            ("Trabajo de las cargas externas",
             f"{fmt(external, 1)} {self.units.energy}"),
            ("Energía cinética en home",
             f"{fmt(trajectory.kinetic_energy[-1], 1)} {self.units.energy}"),
            ("Variación del total (error numérico)",
             f"{drift:.2e} {self.units.energy}"),
        ]
        self.draw_table(fig, 0.38, "Balance", rows)
        return fig


def fmt(value: float, decimals: int = 0) -> str:
    """Número con separador de miles en espacio fino (estilo ES)."""
    return f"{value:,.{decimals}f}".replace(",", " ")


def end_labels(
    ax: Axes, x: float, items: list[tuple[str, float]], gap: float = 0.045
) -> None:
    """Etiquetas al final de cada serie, separadas para no solaparse."""
    lo, hi = ax.get_ylim()
    min_gap = gap * (hi - lo)
    ordered = sorted(items, key=lambda item: item[1])
    placed: list[float] = []
    for _, y in ordered:
        placed.append(y if not placed else max(y, placed[-1] + min_gap))
    for (label, _), y in zip(ordered, placed):
        ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=INK, va="center",
                    annotation_clip=False)


def zigzag(
    p: np.ndarray, q: np.ndarray, turns: int, width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Polilínea en zigzag entre p y q para dibujar un muelle."""
    d = q - p
    n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-12)
    s = np.concatenate(([0.0, 0.1], np.linspace(0.1, 0.9, 2 * turns + 1)[1:],
                        [1.0]))
    offset = np.zeros_like(s)
    offset[2:-2] = width * np.where(np.arange(len(s) - 4) % 2 == 0, 1, -1)
    points = p[None, :] + s[:, None] * d[None, :] + offset[:, None] * n
    return points[:, 0], points[:, 1]
