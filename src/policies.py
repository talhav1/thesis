"""Design policies.

Every policy here is **ignorable** in the sense of the Phase 0 proposition:
the conditional law of X_i given the history H_{i-1} does not depend on the
unknown response parameter theta.  Each policy therefore exposes

    select(state, rng) -> (x, info)
    log_prob(x, state) -> log q_i(x | H_{i-1})

so that the likelihood-factorisation invariant can be checked numerically for
both deterministic and randomised designs.

`OraclePolicy` is the deliberate counterexample: it conditions on the true
theta and so is *not* ignorable.  It exists purely so the test suite can show
that the invariant has real content -- a test that only ever passes is not a
test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.stats import norm

from .response_models import (
    PROB_FLOOR,
    binary_entropy,
    derived_values,
    observed_information,
)
from .utilities import (
    mutual_information_scalar,
    mutual_information_scalar_presorted,
    mutual_information_vector,
    mutual_information_vector_T,
    response_probability_matrix,
    response_probability_matrix_T,
)


@dataclass
class PolicyState:
    """Everything a policy may condition on: the observable history only.

    Also owns a generation-keyed cache.  Between rejuvenations the particle
    *values* are fixed and only their weights change, so the (N, L) response
    probability matrix and its entrywise binary entropy -- by far the most
    expensive objects in a step -- can be reused.  A typical n = 50 run
    rejuvenates about ten times, so this removes roughly 80% of the cost of a
    mutual-information design without changing any number it produces.
    """

    candidates: np.ndarray
    posterior: object                 # ParticlePosterior
    x_hist: list = field(default_factory=list)
    y_hist: list = field(default_factory=list)
    step: int = 0
    _cache: dict = field(default_factory=dict, repr=False)
    _cache_generation: int = field(default=-1, repr=False)

    def _cached(self, key, builder):
        gen = getattr(self.posterior, "generation", 0)
        if gen != self._cache_generation:
            self._cache.clear()
            self._cache_generation = gen
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    def probs_T(self):
        """(L, N) response-probability matrix Phi((x_l - mu_j)/sigma_j)."""
        return self._cached(
            "P_T",
            lambda: response_probability_matrix_T(
                self.candidates, self.posterior.mu, self.posterior.sigma
            ),
        )

    def probs(self):
        """(N, L) view, for code that follows the thesis' index convention."""
        return self.probs_T().T

    def prob_entropies_T(self):
        """(L, N) entrywise binary entropy h(p_lj)."""
        return self._cached("H_T", lambda: binary_entropy(self.probs_T()))

    def sorted_by(self, quantity: str):
        """theta-sorted view of (theta, P_T) for the scalar-MI quadrature."""

        def build():
            theta = derived_values(self.posterior.mu, self.posterior.sigma, quantity)
            order = np.argsort(theta, kind="stable")
            return order, theta[order], np.ascontiguousarray(self.probs_T()[:, order])

        return self._cached(f"sorted::{quantity}", build)

    def invalidate(self):
        """Weights changed; value-keyed caches remain valid."""
        return None


class Policy(Protocol):
    name: str

    def select(self, state: PolicyState, rng: np.random.Generator): ...

    def log_prob(self, x: float, state: PolicyState) -> float: ...


# --------------------------------------------------------------------------
# Deterministic argmax helper
# --------------------------------------------------------------------------


def _argmax_select(scores, candidates):
    idx = int(np.argmax(scores))
    return float(candidates[idx]), idx


def _deterministic_log_prob(x, chosen):
    return 0.0 if np.isclose(x, chosen) else -np.inf


# --------------------------------------------------------------------------
# Rotem's entropy / mutual-information designs
# --------------------------------------------------------------------------


class EntropyVectorPolicy:
    """Maximise I((mu, sigma); Y_{n+1} | x).  Rotem's 'entropy of (mu, sigma)'."""

    name = "entropy_vector"
    target = "(mu,sigma)"

    def scores(self, state):
        return mutual_information_vector_T(
            state.posterior.w, state.probs_T(), state.prob_entropies_T()
        )

    def select(self, state, rng):
        mi = self.scores(state)
        x, idx = _argmax_select(mi, state.candidates)
        return x, {"mi_max": float(mi[idx]), "mi": mi}

    def log_prob(self, x, state):
        mi = self.scores(state)
        return _deterministic_log_prob(x, state.candidates[int(np.argmax(mi))])


class EntropyScalarPolicy:
    """Maximise I(theta; Y_{n+1} | x) for a scalar functional theta.

    `quantity` is one of "mu", "sigma", or "q<p>" (e.g. "q0.95").
    """

    def __init__(self, quantity: str, R: int = 100, c_fraction: float = 0.1,
                 clip_negative: bool = True):
        self.quantity = quantity
        self.R = R
        self.c_fraction = c_fraction
        # Phase 3 default: clip the estimator at zero (protocol section 8).
        # Set False to reproduce Rotem's original unclipped convention.
        self.clip_negative = clip_negative
        self.name = f"entropy_{quantity}"
        self.target = quantity

    def _scores(self, state):
        order, th_sorted, P_sorted = state.sorted_by(self.quantity)
        return mutual_information_scalar_presorted(
            th_sorted, state.posterior.w[order], P_sorted,
            R=self.R, c_fraction=self.c_fraction,
            clip_negative=self.clip_negative,
        )

    def select(self, state, rng):
        res = self._scores(state)
        x, idx = _argmax_select(res.mi, state.candidates)
        info = {"mi_max": float(res.mi[idx]), "mi": res.mi}
        info.update(res.diagnostics())
        return x, info

    def log_prob(self, x, state):
        res = self._scores(state)
        return _deterministic_log_prob(x, state.candidates[int(np.argmax(res.mi))])


# --------------------------------------------------------------------------
# Dror-Steinberg Bayesian D-optimal design
# --------------------------------------------------------------------------


def _fisher_terms(x, mu, sigma):
    """Per-observation information weights v and standardised stresses z.

    Broadcasts x (L,) against mu, sigma (N,) to give (N, L) arrays.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    z = (x[None, :] - mu[:, None]) / sigma[:, None]
    p = np.clip(norm.cdf(z), PROB_FLOOR, 1 - PROB_FLOOR)
    v = norm.pdf(z) ** 2 / (p * (1 - p)) / sigma[:, None] ** 2
    return v, z


def _info_moments(x, mu, sigma):
    """(i11, i12, i22) of the accumulated information, summed over x."""
    if len(np.atleast_1d(x)) == 0:
        n = len(np.atleast_1d(mu))
        return np.zeros(n), np.zeros(n), np.zeros(n)
    v, z = _fisher_terms(x, mu, sigma)
    return v.sum(axis=1), (v * z).sum(axis=1), (v * z * z).sum(axis=1)


def _logdet(i11, i12, i22, floor=1e-300):
    det = i11 * i22 - i12**2
    return np.log(np.maximum(det, floor))


class DrorSteinbergPolicy:
    """Bayesian D-optimality with an m-run Fedorov-style augmentation.

    Follows Dror & Steinberg (2008) as described in Rotem section 3:
    phi2 (a single evaluation at the posterior median) drives the search for
    the m-run augmentation, and phi1 (the posterior average of log|I|) picks
    the single point actually run, from the candidate set formed by the m
    augmentation points plus their median.

    Discrepancy D2: Rotem calls AlgDesign's implementation of Fedorov's
    algorithm.  This is an independent exchange implementation over the same
    candidate grid, so exact numerical agreement with her tables is not
    expected -- only agreement in the operating characteristics.
    """

    name = "dror_steinberg"
    target = "(mu,sigma)"

    def __init__(self, m: int = 4, n_restarts: int = 2, max_sweeps: int = 8):
        self.m = m
        self.n_restarts = n_restarts
        self.max_sweeps = max_sweeps

    def _augmentation(self, state, mu_med, sig_med, rng):
        cand = state.candidates
        L = len(cand)
        v_all, z_all = _fisher_terms(cand, np.array([mu_med]), np.array([sig_med]))
        v_all, z_all = v_all[0], z_all[0]
        c11, c12, c22 = v_all, v_all * z_all, v_all * z_all * z_all

        b11, b12, b22 = _info_moments(np.asarray(state.x_hist), np.array([mu_med]),
                                      np.array([sig_med]))
        b11, b12, b22 = float(b11[0]), float(b12[0]), float(b22[0])

        best_val, best_set = -np.inf, None
        for restart in range(self.n_restarts):
            if restart == 0:
                idx = np.linspace(0, L - 1, self.m + 2)[1:-1].astype(int)
            else:
                idx = rng.choice(L, size=self.m, replace=False)
            idx = np.asarray(idx)
            for _ in range(self.max_sweeps):
                changed = False
                for slot in range(self.m):
                    others = np.delete(idx, slot)
                    a11 = b11 + c11[others].sum()
                    a12 = b12 + c12[others].sum()
                    a22 = b22 + c22[others].sum()
                    vals = _logdet(a11 + c11, a12 + c12, a22 + c22)
                    new = int(np.argmax(vals))
                    if new != idx[slot]:
                        idx[slot] = new
                        changed = True
                if not changed:
                    break
            val = _logdet(
                b11 + c11[idx].sum(), b12 + c12[idx].sum(), b22 + c22[idx].sum()
            )
            if val > best_val:
                best_val, best_set = float(val), idx.copy()
        return np.sort(state.candidates[best_set])

    def _candidate_set(self, aug):
        med = float(np.median(aug))
        return np.concatenate([aug, [med]])

    def _phi1(self, state, base_design, candidate_points):
        post = state.posterior
        b11, b12, b22 = _info_moments(base_design, post.mu, post.sigma)
        v, z = _fisher_terms(candidate_points, post.mu, post.sigma)
        c11, c12, c22 = v, v * z, v * z * z
        ld = _logdet(b11[:, None] + c11, b12[:, None] + c12, b22[:, None] + c22)
        return post.w @ ld

    def select(self, state, rng):
        post = state.posterior
        mu_med = post.median_rotem("mu")
        sig_med = post.median_rotem("sigma")
        aug = self._augmentation(state, mu_med, sig_med, rng)
        cand_set = self._candidate_set(aug)

        design = np.asarray(state.x_hist, dtype=float)
        info_med = observed_information(design, np.asarray(state.y_hist), mu_med, sig_med)
        singular = np.linalg.det(info_med) <= 0 or len(design) < 2

        base = np.concatenate([design, aug]) if singular else design
        scores = self._phi1(state, base, cand_set)
        idx = int(np.argmax(scores))
        return float(cand_set[idx]), {
            "ds_singular_branch": bool(singular),
            "ds_candidate_set": cand_set,
            "phi1": float(scores[idx]),
        }

    def log_prob(self, x, state):
        # deterministic given (history, rng-free path); the augmentation search
        # uses rng only for restarts, which is why replay uses a fixed stream
        raise NotImplementedError(
            "DS augmentation search consumes randomness; use a seeded replay "
            "for factorisation checks instead"
        )


# --------------------------------------------------------------------------
# Non-adaptive and reference designs
# --------------------------------------------------------------------------


class BrucetonPolicy:
    """Dixon-Mood up-and-down.

    Not grid-snapped: Rotem starts from x0 ~ N(mu0, sigma) and moves in steps
    of d, so the realised stresses form their own lattice.  Recorded as D4.
    """

    name = "bruceton"
    target = "(mu,sigma)"

    def __init__(self, step: float, x0: float):
        self.step = step
        self.x0 = x0

    def select(self, state, rng):
        if not state.x_hist:
            return float(self.x0), {}
        last_x, last_y = state.x_hist[-1], state.y_hist[-1]
        return float(last_x - self.step if last_y == 1 else last_x + self.step), {}

    def log_prob(self, x, state):
        chosen, _ = self.select(state, None)
        return _deterministic_log_prob(x, chosen)


class FixedDesignPolicy:
    """Non-adaptive reference design: a fixed set of levels, cycled.

    Chosen once from the prior and never updated, so it is the cleanest
    contrast for attributing a failure to *adaptivity*: it sees the same
    horizon and the same stimulus range but never lets the fitted model steer
    the experiment.
    """

    name = "fixed_design"
    target = None

    def __init__(self, levels=None, mu0: float = 30.0, sigma_hat: float = 2.4,
                 offsets=(-1.5, -0.75, 0.0, 0.75, 1.5)):
        self.levels = (np.asarray(levels, dtype=float) if levels is not None
                       else mu0 + np.asarray(offsets) * sigma_hat)

    def select(self, state, rng):
        return float(self.levels[len(state.x_hist) % len(self.levels)]), {}

    def log_prob(self, x, state):
        return _deterministic_log_prob(x, self.levels[len(state.x_hist) % len(self.levels)])


class UniformGridPolicy:
    """Randomised, non-adaptive: X_i ~ Uniform(candidate grid).

    The simplest ignorable *randomised* policy, and the one used for the
    non-adaptive SBC sanity check (required test 8).
    """

    name = "uniform_grid"
    target = None

    def select(self, state, rng):
        idx = int(rng.integers(len(state.candidates)))
        return float(state.candidates[idx]), {}

    def log_prob(self, x, state):
        L = len(state.candidates)
        return -np.log(L) if np.any(np.isclose(state.candidates, x)) else -np.inf


class SoftmaxEntropyPolicy:
    """Randomised *adaptive* policy: softmax over the vector mutual information.

    Ignorable but non-degenerate, so the factorisation invariant is exercised
    with a genuinely stochastic q_i(. | H_{i-1}) rather than a point mass.
    """

    name = "softmax_entropy"
    target = "(mu,sigma)"

    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature

    def _logits(self, state):
        mi = mutual_information_vector_T(
            state.posterior.w, state.probs_T(), state.prob_entropies_T()
        )
        logits = mi / self.temperature
        return logits - logits.max()

    def select(self, state, rng):
        logits = self._logits(state)
        p = np.exp(logits)
        p /= p.sum()
        idx = int(rng.choice(len(p), p=p))
        return float(state.candidates[idx]), {"select_prob": float(p[idx])}

    def log_prob(self, x, state):
        logits = self._logits(state)
        logZ = np.log(np.exp(logits).sum())
        matches = np.isclose(state.candidates, x)
        if not np.any(matches):
            return -np.inf
        return float(logits[int(np.argmax(matches))] - logZ)


class ExplorationMixturePolicy:
    """(1 - eps) * argmax-MI  +  eps * uniform reference.

    Phase 6 belongs to the safeguard study; this is included now only because
    it is the natural randomised adaptive policy for the ignorability tests,
    and because having it in place early costs nothing.  It is NOT run in any
    Phase 0-2 experiment.
    """

    name = "exploration_mixture"
    target = "(mu,sigma)"

    def __init__(self, eps: float = 0.1, base: Policy | None = None):
        self.eps = eps
        self.base = base or EntropyVectorPolicy()

    def select(self, state, rng):
        if rng.random() < self.eps:
            idx = int(rng.integers(len(state.candidates)))
            return float(state.candidates[idx]), {"explored": True}
        x, info = self.base.select(state, rng)
        info["explored"] = False
        return x, info

    def log_prob(self, x, state):
        L = len(state.candidates)
        base_choice = state.candidates[
            int(np.argmax(mutual_information_vector_T(
                state.posterior.w, state.probs_T(), state.prob_entropies_T())))
        ]
        p = self.eps / L + (1 - self.eps) * float(np.isclose(x, base_choice))
        return float(np.log(p)) if p > 0 else -np.inf


class OraclePolicy:
    """NOT ignorable.  Selects the stimulus nearest the true q_p.

    The negative control for the likelihood-factorisation test: because
    q_i(x | H_{i-1}, theta) depends on theta, the design term does not drop out
    of the posterior and the product likelihood is no longer proportional to
    the full-history likelihood.
    """

    name = "oracle_nonignorable"
    target = None

    def __init__(self, true_curve, p: float = 0.5, sharpness: float = 4.0):
        self.true_curve = true_curve
        self.p = p
        self.sharpness = sharpness

    def set_theta(self, mu, sigma):
        """Re-point the oracle at a candidate theta.

        Used by `factorization.replay_policy_logprobs`: it is precisely this
        method's existence that makes q_i depend on theta and so breaks the
        factorisation, which is what the negative-control test detects.
        """
        from .response_models import ProbitCurve

        self.true_curve = ProbitCurve(float(mu), float(sigma))

    def _logits(self, state):
        target = float(self.true_curve.quantile(self.p))
        return -self.sharpness * np.abs(state.candidates - target)

    def select(self, state, rng):
        logits = self._logits(state)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        idx = int(rng.choice(len(p), p=p))
        return float(state.candidates[idx]), {}

    def log_prob(self, x, state):
        logits = self._logits(state)
        logZ = np.log(np.exp(logits - logits.max()).sum()) + logits.max()
        matches = np.isclose(state.candidates, x)
        if not np.any(matches):
            return -np.inf
        return float(logits[int(np.argmax(matches))] - logZ)


def build_policy(name: str, **kwargs) -> Policy:
    """Factory used by config files, so experiments are fully declarative."""
    table = {
        "entropy_vector": lambda: EntropyVectorPolicy(),
        "entropy_mu": lambda: EntropyScalarPolicy("mu", **kwargs),
        "entropy_sigma": lambda: EntropyScalarPolicy("sigma", **kwargs),
        "entropy_q0.95": lambda: EntropyScalarPolicy("q0.95", **kwargs),
        "dror_steinberg": lambda: DrorSteinbergPolicy(**kwargs),
        "bruceton": lambda: BrucetonPolicy(**kwargs),
        "uniform_grid": lambda: UniformGridPolicy(),
        "fixed_design": lambda: FixedDesignPolicy(**kwargs),
        "softmax_entropy": lambda: SoftmaxEntropyPolicy(**kwargs),
        "exploration_mixture": lambda: ExplorationMixturePolicy(**kwargs),
    }
    if name not in table:
        raise ValueError(f"unknown policy {name!r}; known: {sorted(table)}")
    return table[name]()
