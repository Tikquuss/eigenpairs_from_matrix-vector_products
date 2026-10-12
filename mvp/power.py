"""Power iteration, shifts, deflation and sign separation (Section 3 of the post).

``apply_a(x)`` must return ``A @ x`` for a real symmetric matrix ``A``.
"""

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

MatVec = Callable[[np.ndarray], np.ndarray]


def project_out(basis: np.ndarray, x: np.ndarray, passes: int = 2) -> np.ndarray:
    """Return ``P x = x - Q (Q^T x)``, applied ``passes`` times (reorthogonalization)."""
    if basis.shape[1] == 0:
        return x
    for _ in range(passes):
        x = x - basis @ (basis.T @ x)
    return x


def power_iteration(apply_a: MatVec, v0: np.ndarray, steps: int):
    """Plain power iteration (Proposition 1).

    Returns the Rayleigh quotients ``v_k^T A v_k`` and the residuals
    ``||A v_k - (v_k^T A v_k) v_k||`` for k = 0, ..., steps-1, and the iterates
    v_0, ..., v_{steps-1} as columns. Entry k costs k+1 products.
    """
    v = v0 / np.linalg.norm(v0)
    rayleigh, residual = np.empty(steps), np.empty(steps)
    iterates = np.empty((v.size, steps))
    for k in range(steps):
        iterates[:, k] = v
        av = apply_a(v)
        rayleigh[k] = v @ av
        residual[k] = np.linalg.norm(av - rayleigh[k] * v)
        v = av / np.linalg.norm(av)
    return rayleigh, residual, iterates


@dataclass
class Stage:
    """Result of one deflation stage.

    ``products[i]`` is the number of products used when the candidate had the
    residual ``residuals[i]`` with the original matrix A, and
    ``deflated_residuals[i]`` with the deflated matrix B = P A P.
    ``inherited[i]`` is ||Q^T A q||, the part of the original residual that
    lies in the span of the stored vectors, so that
    residuals[i]**2 = deflated_residuals[i]**2 + inherited[i]**2.
    """

    eigenvalue: float
    vector: np.ndarray
    accepted: bool
    products: List[int] = field(default_factory=list)
    residuals: List[float] = field(default_factory=list)
    deflated_residuals: List[float] = field(default_factory=list)
    inherited: List[float] = field(default_factory=list)

    @property
    def cost(self) -> int:
        return self.products[-1] if self.products else 0


def _accepted(residual: float, aq: np.ndarray, eps_abs: float, eps_rel: float) -> bool:
    return residual <= eps_abs + eps_rel * np.linalg.norm(aq)


def _record(stage: Stage, products: int, q: np.ndarray, aq: np.ndarray, basis: np.ndarray):
    """Append the residuals of the candidate q with A and with B = P A P."""
    lam = q @ aq
    stage.eigenvalue, stage.vector = lam, q
    stage.products.append(products)
    stage.residuals.append(np.linalg.norm(aq - lam * q))
    bq = project_out(basis, aq)                     # B q = P A q, since P q = q
    stage.deflated_residuals.append(np.linalg.norm(bq - (q @ bq) * q))
    stage.inherited.append(np.linalg.norm(basis.T @ aq) if basis.shape[1] else 0.0)
    return stage.residuals[-1]


def _stop(stage: Stage, aq: np.ndarray, eps_abs: float, eps_rel: float, limit: float) -> bool:
    """Stopping rule of a stage. Without ``limit``, the test of Algorithm 1 on
    the original residual. With ``limit``, the stage runs until the residual
    for the deflated matrix B falls below ``limit`` (the numerical counterpart
    of taking the inner iteration to its limit, as in Corollary 2), and the
    original residual is then checked against the tolerance."""
    if limit is None:
        stage.accepted = _accepted(stage.residuals[-1], aq, eps_abs, eps_rel)
        return stage.accepted
    if stage.deflated_residuals[-1] <= limit:
        stage.accepted = _accepted(stage.residuals[-1], aq, eps_abs, eps_rel)
        return True
    return False


def algorithm1_stage(apply_a: MatVec, basis: np.ndarray, w0: np.ndarray, eps_abs: float = 1e-10,
                     eps_rel: float = 1e-10, max_steps: int = 5000, limit: float = None) -> Stage:
    """One stage of Algorithm 1: squared power iteration with sign separation on
    ``B = P A P``, where ``P`` projects out the columns of ``basis``.

    Each step uses two products, ``h1 = A w`` and ``h2 = A y``; the candidate's
    product ``A q = (h1 + s h2) / n`` comes for free by linearity. The candidate
    is accepted when its residual with the original matrix satisfies
    ``||A q - lam q|| <= eps_abs + eps_rel ||A q||`` (see ``_stop`` for ``limit``).
    """
    w = project_out(basis, w0)
    w = w / np.linalg.norm(w)
    stage = Stage(0.0, w, False)
    products = 0
    for _ in range(max_steps):
        h1 = apply_a(w)                                   # step 1: first product
        products += 1
        a = project_out(basis, h1)
        rho = np.linalg.norm(a)
        if rho == 0.0:                                    # step 5: zero projected product
            _record(stage, products, w, h1, basis)
            stage.accepted = _accepted(stage.residuals[-1], h1, eps_abs, eps_rel)
            return stage
        y = a / rho                                       # step 2: second product
        h2 = apply_a(y)
        products += 1
        b = project_out(basis, h2)
        s = 1.0 if w @ y >= 0 else -1.0                   # step 3: candidate
        n = np.linalg.norm(w + s * y)
        q = (w + s * y) / n
        aq = (h1 + s * h2) / n                            # A q by linearity
        _record(stage, products, q, aq, basis)
        if _stop(stage, aq, eps_abs, eps_rel, limit):     # step 4: check or update
            return stage
        w = b / np.linalg.norm(b)
    return stage


def deflated_power_stage(apply_a: MatVec, basis: np.ndarray, w0: np.ndarray, shift: float = 0.0,
                         eps_abs: float = 1e-10, eps_rel: float = 1e-10,
                         max_products: int = 5000, limit: float = None) -> Stage:
    """Ordinary power iteration on ``P (A - shift I) P`` with the stopping rule
    of Algorithm 1, the residual being computed with the original matrix A.

    With ``shift = 0`` this is plain deflated power iteration: it converges when
    one remaining eigenvalue strictly dominates in absolute value
    (Proposition 1) and stalls when +rho and -rho tie (Remark 5). With
    ``shift <= lambda_min(A)`` every eigenvalue of A - shift I is nonnegative,
    and each stage finds the largest remaining eigenvalue of A (Corollary 1).
    """
    w = project_out(basis, w0)
    w = w / np.linalg.norm(w)
    stage = Stage(0.0, w, False)
    for products in range(1, max_products + 1):
        h = apply_a(w)
        _record(stage, products, w, h, basis)
        if _stop(stage, h, eps_abs, eps_rel, limit):
            return stage
        a = project_out(basis, h - shift * w)
        w = a / np.linalg.norm(a)
    return stage


def deflated_eigenpairs(apply_a: MatVec, dimension: int, count: int, rng: np.random.Generator,
                        stage=algorithm1_stage, **stage_options) -> List[Stage]:
    """Corollary 2: ``count`` eigenpairs, one per deflation stage, each stage
    started from a fresh Gaussian vector projected on the orthogonal complement
    of the vectors already accepted. Stops early if a stage is not accepted."""
    basis = np.zeros((dimension, 0))
    stages = []
    for _ in range(count):
        result = stage(apply_a, basis, rng.standard_normal(dimension), **stage_options)
        stages.append(result)
        if not result.accepted:
            break
        q = project_out(basis, result.vector)
        basis = np.column_stack([basis, q / np.linalg.norm(q)])
    return stages
