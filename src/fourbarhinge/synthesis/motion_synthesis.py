"""
Síntesis dimensional por guiado del acoplador (generación de movimiento).

Dadas dos trayectorias sincronizadas P_i, Q_i (ejes globales) de dos
puntos del acoplador, busca la geometría del cuadrilátero que mejor lleva
el acoplador por esas posiciones (mínimos cuadrados si hay muchas).

Cada posición i fija el acoplador entero (origen P_i, eje x hacia Q_i).
El mecanismo se separa en dos díadas independientes:
    entrada: A (fijo) - B (en el acoplador), |B_i - A| = Li
    salida:  D (fijo) - C (en el acoplador), |C_i - D| = Lo

1. Candidatos de díada: para cada punto b del acoplador (rejilla en el
   marco P-Q), el mejor pivote fijo es el centro de la circunferencia
   ajustada a sus posiciones b_i. Con A fijo, b es el centro de la
   circunferencia ajustada a A visto desde el acoplador (inversión).
2. Se combinan pares de díadas y se pule el mecanismo completo por
   mínimos cuadrados sobre el error de posición de P y Q, penalizando que
   no pueda montarse entre posiciones o un ángulo de transmisión pequeño.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares

from ..models.barra import Point
from .path_synthesis import (
    LinkBounds, PathSynthesisResult, coupler_point_positions,
    transmission_angle,
)


def rotate(angle: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Gira local (2,) o (m,2) los ángulos angle (n,) -> (n,2) o (m,n,2)."""
    c, s = np.cos(angle), np.sin(angle)
    x, y = local[..., 0:1], local[..., 1:2]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def rotate_each(angle: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Gira cada vector (n,2) su ángulo (n,)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * vector[:, 0] - s * vector[:, 1],
                     s * vector[:, 0] + c * vector[:, 1]], axis=1)


def fit_circles(points: np.ndarray, weights: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ajuste algebraico (Kåsa) de circunferencias, vectorizado.
    points (m, n, 2) -> centro (m, 2), radio (m,), error RMS radial (m,).
    """
    x, y = points[..., 0], points[..., 1]
    w = weights / weights.sum()
    m = np.stack([x, y, np.ones_like(x)], axis=-1)            # (m, n, 3)
    rhs = -(x**2 + y**2)
    mtm = np.einsum("kni,n,knj->kij", m, w, m)
    mtr = np.einsum("kni,n,kn->ki", m, w, rhs)
    mtm += 1e-12 * np.eye(3) * np.trace(mtm, axis1=1, axis2=2)[:, None, None]
    a = np.linalg.solve(mtm, mtr[..., None])[..., 0]
    center = -0.5 * a[:, :2]
    radius = np.sqrt(np.maximum((center**2).sum(axis=1) - a[:, 2], 0.0))
    distance = np.linalg.norm(points - center[:, None, :], axis=2)
    error = np.sqrt(np.einsum("n,kn->k", w, (distance - radius[:, None])**2))
    return center, radius, error


@dataclass
class Dyad:
    """Díada: pivote fijo (global) y pivote del acoplador (marco P-Q)."""
    ground: np.ndarray
    moving: np.ndarray
    length: float
    error: float


@dataclass
class MotionSynthesisResult(PathSynthesisResult):
    """PathSynthesisResult de P más el segundo punto Q del acoplador."""
    # Q local en el acoplador
    second_point: Point = field(default_factory=lambda: Point(x=0.0, y=0.0))
    target_points_q: np.ndarray | None = None
    achieved_points_q: np.ndarray | None = None
    rigidity_error: float = 0.0       # máx. variación de |P_i Q_i|
    input_monotonic: bool = True

    def path_q(self, n: int = 200) -> np.ndarray:
        """Trayectoria continua de Q entre la primera y última posición."""
        ti = np.linspace(self.theta_start, self.theta_end, n)
        params = dict(self.params, u=self.second_point.x,
                      v=self.second_point.y)
        return coupler_point_positions(params, ti, self.branch)[0]

    def summary(self) -> str:
        q = self.second_point
        return (super().summary() + "\n"
                f"Q en acoplador = ({q.x:.3f}, {q.y:.3f})  "
                f"variación |PQ| en los datos = {self.rigidity_error:.3g}  "
                f"entrada monótona = {self.input_monotonic}")


class CouplerMotionSynthesis:
    """
    Busca la geometría del cuadrilátero que lleva los puntos P y Q del
    acoplador por track_p[i], track_q[i] (posiciones sincronizadas).

    Con muchas posiciones devuelve el mejor ajuste por mínimos cuadrados;
    weights (n,) da más importancia a algunas (p.ej. cerrado y abierto).
    ground_pivot_a / ground_pivot_d: pivotes fijos A y D (global); D sólo
    se admite junto con A.
    """

    def __init__(
        self,
        track_p: np.ndarray,
        track_q: np.ndarray,
        bounds: LinkBounds,
        weights: np.ndarray | None = None,
        ground_pivot_a: Point | None = None,
        ground_pivot_d: Point | None = None,
        branches: tuple[int, ...] = (1, -1),
        grid: int = 61,
        candidates: int = 6,
        midpoints: int = 4,
    ):
        self.track_p = np.asarray(track_p, dtype=float)
        self.track_q = np.asarray(track_q, dtype=float)
        if (self.track_p.ndim != 2 or self.track_p.shape[1] != 2
                or self.track_p.shape != self.track_q.shape):
            raise ValueError("track_p and track_q must have the same shape "
                             "(n, 2)")
        self.n_poses = len(self.track_p)
        if self.n_poses < 3:
            raise ValueError("At least 3 poses are required")
        if ground_pivot_d is not None and ground_pivot_a is None:
            raise ValueError("ground_pivot_d requires ground_pivot_a")
        self.bounds = bounds
        self.weights = (np.ones(self.n_poses) if weights is None
                        else np.asarray(weights, dtype=float))
        if self.weights.shape != (self.n_poses,) or np.any(self.weights < 0):
            raise ValueError("weights must be (n,) and non-negative")
        self.pivot_a = (None if ground_pivot_a is None
                        else ground_pivot_a.to_array())
        self.pivot_d = (None if ground_pivot_d is None
                        else ground_pivot_d.to_array())
        self.branches = branches
        self.grid = grid
        self.candidates = candidates
        self.midpoints = midpoints

        # Posición del acoplador en cada pose: origen P_i, eje x hacia Q_i
        pq = self.track_q - self.track_p
        distance = np.linalg.norm(pq, axis=1)
        if np.any(distance < 1e-12):
            raise ValueError("P and Q coincide in some pose")
        self.pq_length = float(np.average(distance, weights=self.weights))
        self.rigidity_error = float(np.max(np.abs(distance - self.pq_length)))
        self.pose_angle = np.unwrap(np.arctan2(pq[:, 1], pq[:, 0]))
        points = np.vstack((self.track_p, self.track_q))
        self.scale = max(float(np.ptp(points, axis=0).max()),
                         bounds.min_length)

    # ------------------------------------------------------------------
    # Díadas
    # ------------------------------------------------------------------

    def moving_positions(self, moving: np.ndarray) -> np.ndarray:
        """Punto del acoplador (marco P-Q) -> global en cada pose (n,2)."""
        return self.track_p + rotate(self.pose_angle, moving)

    def dyad_residuals(self, x: np.ndarray, ground: np.ndarray | None
                       ) -> np.ndarray:
        if ground is None:
            ground, moving, length = x[:2], x[2:4], x[4]
        else:
            moving, length = x[:2], x[2]
        distance = np.linalg.norm(self.moving_positions(moving) - ground,
                                  axis=1)
        return np.sqrt(self.weights) * (distance - length) / self.scale

    def refine_dyad(self, ground: np.ndarray, moving: np.ndarray,
                    length: float, fixed_ground: bool) -> Dyad:
        b = self.bounds
        length = float(np.clip(length, b.min_length * 1.001,
                               b.max_length * 0.999))
        if fixed_ground:
            x0, lower, upper = (np.r_[moving, length],
                                [-np.inf, -np.inf, b.min_length],
                                [np.inf, np.inf, b.max_length])
        else:
            x0, lower, upper = (np.r_[ground, moving, length],
                                [-np.inf] * 4 + [b.min_length],
                                [np.inf] * 4 + [b.max_length])
        solution = least_squares(
            self.dyad_residuals, x0, bounds=(lower, upper),
            args=(ground if fixed_ground else None,), max_nfev=200)
        x = solution.x
        if fixed_ground:
            ground, moving, length = ground, x[:2], x[2]
        else:
            ground, moving, length = x[:2], x[2:4], x[4]
        error = float(np.sqrt(np.mean(self.dyad_residuals(
            x, ground if fixed_ground else None)**2)) * self.scale)
        return Dyad(np.array(ground), np.array(moving), float(length), error)

    def fixed_ground_dyad(self, ground: np.ndarray) -> Dyad:
        """Inversión: el pivote fijo visto desde el acoplador describe una
        curva; el pivote del acoplador es el centro de su circunferencia."""
        relative = rotate_each(-self.pose_angle, ground - self.track_p)
        center, radius, _ = fit_circles(relative[None], self.weights)
        return self.refine_dyad(ground, center[0], float(radius[0]), True)

    def free_dyads(self) -> list[Dyad]:
        """Mejores díadas libres: rejilla de puntos del acoplador."""
        span = self.bounds.max_length
        axis = np.linspace(-span, span, self.grid)
        gx, gy = np.meshgrid(axis, axis + 0.0)
        moving = np.stack([gx.ravel(), gy.ravel()], axis=1)       # (m, 2)
        positions = (self.track_p[None] + rotate(self.pose_angle, moving))
        center, radius, error = fit_circles(positions, self.weights)
        ok = ((radius >= self.bounds.min_length)
              & (radius <= self.bounds.max_length) & np.isfinite(error))
        # Con datos reales el error forma valles casi planos: se escogen
        # puntos separados (sin refinar cada uno, que los llevaría al mismo
        # mínimo); el pulido del mecanismo completo los ajusta después.
        order = np.argsort(np.where(ok, error, np.inf))
        separation = max(4.0 * (axis[1] - axis[0]),
                         0.5 * self.bounds.min_length)
        chosen: list[int] = []
        for k in order:
            if not ok[k] or len(chosen) >= self.candidates:
                break
            if all(np.linalg.norm(moving[k] - moving[j]) > separation
                   for j in chosen):
                chosen.append(k)
        return [Dyad(center[k], moving[k], float(radius[k]), float(error[k]))
                for k in chosen]

    # ------------------------------------------------------------------
    # Mecanismo completo
    # ------------------------------------------------------------------

    def initial_mechanism(self, input_dyad: Dyad, output_dyad: Dyad
                          ) -> tuple[dict[str, float], np.ndarray]:
        """Parámetros de la librería y ángulos de entrada de dos díadas."""
        a, d = input_dyad.ground, output_dyad.ground
        b, c = input_dyad.moving, output_dyad.moving
        theta_ground = float(np.arctan2(*(d - a)[::-1]))
        beta = float(np.arctan2(*(c - b)[::-1]))   # eje B->C en marco P-Q
        p_local = rotate(np.array([-beta]), -b)[0]
        b_global = self.moving_positions(b)
        theta = np.unwrap(np.arctan2(b_global[:, 1] - a[1],
                                     b_global[:, 0] - a[0])) - theta_ground
        params = {
            "ax": float(a[0]), "ay": float(a[1]),
            "theta_ground": theta_ground,
            "lg": float(np.linalg.norm(d - a)),
            "li": input_dyad.length, "lc": float(np.linalg.norm(c - b)),
            "lo": output_dyad.length,
            "u": float(p_local[0]), "v": float(p_local[1]),
            "w": -beta,                    # ángulo de P->Q en el acoplador
        }
        return params, theta

    def fixed_parameters(self) -> dict[str, float]:
        fixed: dict[str, float] = {}
        if self.pivot_a is not None:
            fixed["ax"], fixed["ay"] = map(float, self.pivot_a)
        if self.pivot_a is not None and self.pivot_d is not None:
            d = self.pivot_d - self.pivot_a
            fixed["theta_ground"] = float(np.arctan2(d[1], d[0]))
            fixed["lg"] = float(np.linalg.norm(d))
        return fixed

    def points(self, params: dict[str, float], theta: np.ndarray,
               branch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
        """P, Q globales y tc, to, alfa en los ángulos theta."""
        p, tc, to, alfa = coupler_point_positions(params, theta, branch)
        q_local = np.array([params["u"], params["v"]]) + self.pq_length * \
            np.array([np.cos(params["w"]), np.sin(params["w"])])
        q, _, _, _ = coupler_point_positions(
            dict(params, u=q_local[0], v=q_local[1]), theta, branch)
        return p, q, tc, to, alfa

    def sample_angles(self, theta: np.ndarray) -> np.ndarray:
        t = np.linspace(0.0, 1.0, self.midpoints + 2)[1:-1]
        mids = theta[:-1, None] + t[None, :] * np.diff(theta)[:, None]
        return np.concatenate((theta, mids.ravel()))

    def residuals(self, x: np.ndarray, names: list[str],
                  fixed: dict[str, float], branch: int) -> np.ndarray:
        params = dict(fixed)
        params.update(zip(names, x[:len(names)]))
        theta = x[len(names):]
        p, q, _, _, _ = self.points(params, theta, branch)
        w = np.sqrt(self.weights)[:, None]
        position = np.concatenate((
            (w * (p - self.track_p)).ravel(),
            (w * (q - self.track_q)).ravel())) / self.scale
        _, tc, to, alfa = coupler_point_positions(
            params, self.sample_angles(theta), branch)
        assembly = 10.0 * np.maximum(np.abs(alfa) - 1.0, 0.0)
        transmission = np.maximum(
            self.bounds.min_transmission_angle - transmission_angle(tc, to),
            0.0)
        return np.concatenate((position, assembly, transmission))

    def jacobian(self, x: np.ndarray, names: list[str],
                 fixed: dict[str, float], branch: int,
                 upper: np.ndarray) -> np.ndarray:
        """
        Jacobiano por diferencias finitas agrupadas: θ_i y θ_{i+2} no
        comparten residuos, así que los ángulos se perturban en 2 grupos
        (n_params + 2 evaluaciones en vez de n_params + n).
        """
        n_names = len(names)
        f0 = self.residuals(x, names, fixed, branch)
        sparsity = self.jacobian_sparsity(n_names)
        jac = np.zeros((len(f0), len(x)))
        step = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(x))
        step = np.where(x + step > upper, -step, step)
        for j in range(n_names):
            shifted = x.copy()
            shifted[j] += step[j]
            jac[:, j] = (self.residuals(shifted, names, fixed, branch)
                         - f0) / step[j]
        for color in (0, 1):
            columns = n_names + np.arange(color, self.n_poses, 2)
            shifted = x.copy()
            shifted[columns] += step[columns]
            diff = self.residuals(shifted, names, fixed, branch) - f0
            for j in columns:
                jac[sparsity[:, j], j] = diff[sparsity[:, j]] / step[j]
        return jac

    def jacobian_sparsity(self, n_names: int) -> np.ndarray:
        """Cada θ_i sólo afecta a su pose y a las muestras contiguas."""
        n, m = self.n_poses, self.midpoints
        n_samples = n + (n - 1) * m
        rows = 4 * n + 2 * n_samples
        sparsity = np.zeros((rows, n_names + n), dtype=bool)
        sparsity[:, :n_names] = True
        pose = np.arange(n)
        for offset in (0, 2 * n):                       # P y Q
            for k in (0, 1):
                sparsity[offset + 2 * pose + k, n_names + pose] = True
        mids = n + np.arange((n - 1) * m)
        left = np.repeat(np.arange(n - 1), m)
        for offset in (4 * n, 4 * n + n_samples):        # montaje, transmisión
            sparsity[offset + pose, n_names + pose] = True
            sparsity[offset + mids, n_names + left] = True
            sparsity[offset + mids, n_names + left + 1] = True
        return sparsity

    def polish(self, params: dict[str, float], theta: np.ndarray,
               branch: int, max_nfev: int = 2000) -> MotionSynthesisResult:
        fixed = self.fixed_parameters()
        names = [k for k in ("ax", "ay", "theta_ground", "lg", "li", "lc",
                             "lo", "u", "v", "w") if k not in fixed]
        b = self.bounds
        limits = {k: (b.min_length, b.max_length)
                  for k in ("lg", "li", "lc", "lo")}
        lower = np.array([limits.get(k, (-np.inf, np.inf))[0] for k in names]
                         + [-np.inf] * self.n_poses)
        upper = np.array([limits.get(k, (-np.inf, np.inf))[1] for k in names]
                         + [np.inf] * self.n_poses)
        x0 = np.array([params[k] for k in names] + list(theta))
        span = upper - lower
        finite = np.isfinite(span)
        x0[finite] = np.clip(x0[finite], lower[finite] + 1e-6 * span[finite],
                             upper[finite] - 1e-6 * span[finite])
        solution = least_squares(
            self.residuals, x0, bounds=(lower, upper),
            args=(names, fixed, branch), max_nfev=max_nfev, xtol=1e-12,
            ftol=1e-12,
            jac=lambda x, *args: self.jacobian(x, *args, upper=upper))
        params = dict(fixed)
        params.update(zip(names, solution.x[:len(names)]))
        return self.build_result(params, solution.x[len(names):], branch)

    def build_result(self, params: dict[str, float], theta: np.ndarray,
                     branch: int) -> MotionSynthesisResult:
        p, q, _, _, _ = self.points(params, theta, branch)
        steps = np.diff(theta)
        monotonic = bool(np.all(steps > 0) or np.all(steps < 0))
        dense = np.linspace(theta[0], theta[-1], 400)
        _, tc, to, alfa = coupler_point_positions(params, dense, branch)
        mu = transmission_angle(tc, to)
        errors = np.concatenate((np.linalg.norm(p - self.track_p, axis=1),
                                 np.linalg.norm(q - self.track_q, axis=1)))
        feasible = bool(monotonic and np.all(np.abs(alfa) <= 1.0)
                        and mu.min() >= self.bounds.min_transmission_angle
                        - 1e-6)
        q_local = np.array([params["u"], params["v"]]) + self.pq_length * \
            np.array([np.cos(params["w"]), np.sin(params["w"])])
        return MotionSynthesisResult(
            params={k: float(v) for k, v in params.items()},
            branch=branch,
            theta_inputs=np.asarray(theta, dtype=float),
            target_points=self.track_p,
            achieved_points=p,
            rms_error=float(np.sqrt(np.mean(errors**2))),
            max_error=float(errors.max()),
            feasible=feasible,
            min_transmission_angle=float(mu.min()),
            second_point=Point(x=float(q_local[0]), y=float(q_local[1])),
            target_points_q=self.track_q,
            achieved_points_q=q,
            rigidity_error=self.rigidity_error,
            input_monotonic=monotonic,
        )

    # ------------------------------------------------------------------

    def solve(self, polish: int = 8) -> MotionSynthesisResult:
        """
        polish: número de combinaciones de díadas (las de menor error
        inicial) que se pulen con el mecanismo completo.
        """
        free = None
        if self.pivot_a is not None:
            inputs = [self.fixed_ground_dyad(self.pivot_a)]
        else:
            free = self.free_dyads()
            inputs = free
        if self.pivot_d is not None:
            outputs = [self.fixed_ground_dyad(self.pivot_d)]
        else:
            outputs = free if free is not None else self.free_dyads()
        if not inputs or not outputs:
            raise RuntimeError("No dyad fits the motion within the bounds")

        starts = []
        min_length = self.bounds.min_length
        for i in inputs:
            for o in outputs:
                if (np.linalg.norm(i.moving - o.moving) < min_length
                        or np.linalg.norm(i.ground - o.ground) < min_length):
                    continue
                params, theta = self.initial_mechanism(i, o)
                for branch in self.branches:
                    p, q, _, _, _ = self.points(params, theta, branch)
                    error = float(np.sqrt(np.mean(np.concatenate((
                        np.linalg.norm(p - self.track_p, axis=1),
                        np.linalg.norm(q - self.track_q, axis=1)))**2)))
                    starts.append((error, params, theta, branch))
        if not starts:
            raise RuntimeError("No pair of dyads forms a valid mechanism")
        starts.sort(key=lambda s: s[0])

        # Pulido corto de todas las combinaciones y completo de las 2 mejores
        def key(result: MotionSynthesisResult) -> tuple[bool, float]:
            return (result.feasible, -result.rms_error)

        quick = [self.polish(params, theta, branch, max_nfev=100)
                 for _, params, theta, branch in starts[:polish]]
        quick.sort(key=key, reverse=True)
        final = [self.polish(r.params, r.theta_inputs, r.branch)
                 for r in quick[:2]]
        return max(final, key=key)
