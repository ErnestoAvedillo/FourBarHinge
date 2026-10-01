
"""
Gráficos e informe A4 de la síntesis (trayectoria + muelles).
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from ..models.constants import BarId  # noqa: E402
from ..synthesis.path_synthesis import PathSynthesisResult  # noqa: E402
from ..synthesis.spring_synthesis import SpringSynthesisResult  # noqa: E402
from .plots import (  # noqa: E402
    BAR_COLOR, BAR_NAME, INK, INK_2, SURFACE, FourBarPlotter, fmt, footer,
    new_page, style_axes,
)

TARGET_COLOR = "#e34948"
PATH_COLOR = BAR_COLOR[BarId.COUPLER]
EFFORT_COLOR = "#2a78d6"


class SynthesisPlotter:
    def __init__(self, plotter: FourBarPlotter):
        """plotter: FourBarPlotter de la geometría y muelles obtenidos."""
        self.plotter = plotter
        self.units = plotter.units

    # ------------------------------------------------------------------

    def plot_path(self, ax: Axes, result: PathSynthesisResult,
                  theta_input: float | None = None) -> None:
        """Mecanismo + trayectoria obtenida + puntos objetivo."""
        theta = result.theta_start if theta_input is None else theta_input
        self.plotter.plot_geometry(ax, theta,
                                   ghost_theta_input=result.theta_end,
                                   show_actuator=False)
        path = result.path()
        ax.plot(path[:, 0], path[:, 1], color=PATH_COLOR, linewidth=1.2,
                linestyle="-", alpha=0.6, zorder=4,
                label="Trayectoria de P")
        targets = result.target_points
        ax.plot(targets[:, 0], targets[:, 1], linestyle="none", marker="o",
                markersize=8, markerfacecolor="none",
                markeredgecolor=TARGET_COLOR, markeredgewidth=1.5, zorder=9,
                label="Puntos objetivo")
        for k, p in enumerate(targets, start=1):
            ax.annotate(str(k), p, xytext=(6, -10), textcoords="offset points",
                        fontsize=8, color=INK_2, zorder=9)
        ax.plot(*result.achieved_points.T, linestyle="none", marker="o",
                markersize=3.5, color=PATH_COLOR, zorder=10,
                label="P obtenido")
        ax.relim()
        ax.autoscale_view()

    def plot_effort_travel(
        self, ax: Axes, travel: np.ndarray, effort: np.ndarray,
        target: np.ndarray | None = None,
    ) -> None:
        style_axes(ax)
        if target is not None:
            ax.plot(travel, target, color=TARGET_COLOR, linewidth=1.4,
                    label="Objetivo")
        ax.plot(travel, effort, color=EFFORT_COLOR, linewidth=1.8,
                label="Con los muelles")
        ax.axhline(0.0, color=INK_2, linewidth=0.8)
        ax.set_xlabel(f"Recorrido de P [{self.units.length}]")
        ax.set_ylabel(f"Esfuerzo tangente [{self.units.force}]")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left",
                  bbox_to_anchor=(0, 1.0), labelcolor=INK_2)

    # ------------------------------------------------------------------

    def build_report(
        self,
        path: str | Path,
        path_result: PathSynthesisResult,
        spring_result: SpringSynthesisResult | None = None,
        title: str = "Síntesis de bisagra de 4 barras",
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pages = [self.page_path(path_result, title)]
        if spring_result is not None:
            pages.append(self.page_springs(spring_result))
        with PdfPages(path) as pdf:
            for number, fig in enumerate(pages, start=1):
                footer(fig, number)
                pdf.savefig(fig)
                plt.close(fig)
            pdf.infodict()["Title"] = title
        return path

    def page_path(self, result: PathSynthesisResult, title: str) -> Figure:
        fig = new_page(title, "Síntesis por trayectoria del punto P del "
                              "acoplador. En gris, la posición final.")
        ax = fig.add_axes((0.1, 0.47, 0.82, 0.42))
        self.plot_path(ax, result)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=7.5,
                  ncol=4, loc="upper left", bbox_to_anchor=(0, -0.09),
                  labelcolor=INK_2)
        u = self.units.length
        p = result.params
        rows = [
            ("Parámetro", "Valor"),
            ("Pivote A", f"({p['ax']:.2f}, {p['ay']:.2f}) {u}"),
            ("Orientación barra fija", f"{np.rad2deg(p['theta_ground']):.2f}°"),
            (f"{BAR_NAME[BarId.GROUND]} / {BAR_NAME[BarId.INPUT]}",
             f"{p['lg']:.2f} / {p['li']:.2f} {u}"),
            (f"{BAR_NAME[BarId.COUPLER]} / {BAR_NAME[BarId.OUTPUT]}",
             f"{p['lc']:.2f} / {p['lo']:.2f} {u}"),
            ("P en el acoplador (local)", f"({p['u']:.2f}, {p['v']:.2f}) {u}"),
            ("Rama de montaje", f"{result.branch:+d}"),
            ("θ entrada", f"{np.rad2deg(result.theta_start):.1f}° → "
                          f"{np.rad2deg(result.theta_end):.1f}°"),
            ("Error RMS / máximo",
             f"{result.rms_error:.3g} / {result.max_error:.3g} {u}"),
            ("Ángulo de transmisión mínimo",
             f"{np.rad2deg(result.min_transmission_angle):.1f}°"),
            ("Montable en todo el recorrido",
             "sí" if result.feasible else "NO"),
        ]
        self.plotter.draw_table(fig, 0.36, "Geometría obtenida", rows)
        return fig

    def page_springs(self, result: SpringSynthesisResult) -> Figure:
        fig = new_page(
            "Curva esfuerzo – recorrido",
            "Fuerza estática en P, tangente a su trayectoria. "
            "F > 0: hay que empujar en el sentido del recorrido.")
        ax = fig.add_axes((0.12, 0.55, 0.8, 0.32))
        target = None if np.all(np.isnan(result.target)) else result.target
        self.plot_effort_travel(ax, result.travel, result.effort, target)
        u = self.units
        rows = [("Muelle", "Constante", "Libre", "Anclajes (local)")]
        for k, s in enumerate(result.compression_springs, start=1):
            a, b = s.a.position.to_array(), s.b.position.to_array()
            rows.append((
                f"C{k} {BAR_NAME[BarId(s.a.bar)]}–{BAR_NAME[BarId(s.b.bar)]}",
                f"{fmt(s.constant, 2)} {u.force}/{u.length}",
                f"{s.free_length:.2f} {u.length}",
                f"({a[0]:.1f}, {a[1]:.1f}) – ({b[0]:.1f}, {b[1]:.1f})"))
        for k, s in enumerate(result.torsion_springs, start=1):
            rows.append((
                f"T{k} {BAR_NAME[BarId(s.bar_a)]}–{BAR_NAME[BarId(s.bar_b)]}",
                f"{fmt(s.constant, 1)} {u.torque}/rad",
                f"{np.rad2deg(s.free_angle):.1f}°", "en la bisagra"))
        y = self.plotter.draw_table(fig, 0.44, "Muelles", rows)
        fig.text(0.08, y - 0.03, "Rangos de trabajo", fontsize=12,
                 fontweight="bold", color=INK)
        for k, line in enumerate(result.spring_ranges):
            fig.text(0.08, y - 0.065 - 0.024 * k, line, fontsize=8.5,
                     color=INK)
        if not np.all(np.isnan(result.target)):
            fig.text(0.08, y - 0.075 - 0.024 * len(result.spring_ranges),
                     f"Error RMS respecto al objetivo: "
                     f"{fmt(result.rms_error, 1)} {u.force} "
                     f"(máx {fmt(result.max_error, 1)} {u.force})",
                     fontsize=8.5, color=INK_2)
        fig.patch.set_facecolor(SURFACE)
        return fig
