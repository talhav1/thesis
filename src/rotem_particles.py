"""Rotem's weighted-particle posterior, reimplemented faithfully.

The algorithm (thesis sections 4.1-4.2 and 5.4):

1. draw N = 10,000 parameter vectors from the prior;
2. weight each by the normalised likelihood of the whole history so far;
3. after each new observation, measure D_KL(W^(k) || W^(k-1));
4. if that exceeds 0.1, replace the particle set by a draw from
   N(beta_med, Sigma_hat) in the GLM parameterisation and reweight by the
   *adjusted* likelihood L * pi_0(beta) / pi_new(beta).

Deliberate implementation choices, all recorded in `docs/discrepancies.md`:

* weights are carried in log space (`logsumexp` normalisation).  Rotem's
  written formulae normalise raw likelihoods, which underflows to 0/0 for
  n >~ 40 in double precision.  This is a pure numerical fix; the represented
  posterior is identical wherever her version does not underflow (D5).
* after a rejuvenation the KL baseline is reset to the new particle set,
  because a KL divergence between weight vectors on *different* supports is
  undefined.  The thesis does not specify this (D6).
* particles drawn with beta1 <= 0 (i.e. sigma <= 0) receive prior density 0
  and hence weight 0.  They are counted, not discarded silently (D7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from .posterior_grid import lower_inverse_cdf, weighted_quantile
from .response_models import derived_values, probit_loglik

KL_THRESHOLD = 0.1
DEFAULT_N_PARTICLES = 10_000


@dataclass
class ParticleDiagnostics:
    ess: float = float("nan")
    max_weight: float = float("nan")
    kl: float = float("nan")
    n_resamples: int = 0
    n_invalid_particles: int = 0
    resample_steps: list = field(default_factory=list)
    kl_trace: list = field(default_factory=list)
    ess_trace: list = field(default_factory=list)
    covariance_jitter: int = 0

    def as_dict(self):
        return {
            "ess": self.ess,
            "max_weight": self.max_weight,
            "kl_last": self.kl,
            "n_resamples": self.n_resamples,
            "n_invalid_particles": self.n_invalid_particles,
            "resample_steps": list(self.resample_steps),
            "covariance_jitter": self.covariance_jitter,
        }


class ParticlePosterior:
    """Weighted discrete representation of the (mu, sigma) posterior."""

    def __init__(self, prior, n_particles: int, rng: np.random.Generator,
                 kl_threshold: float = KL_THRESHOLD, rejuvenate: bool = True):
        self.prior = prior
        self.rng = rng
        self.n = n_particles
        self.kl_threshold = kl_threshold
        self.rejuvenate = rejuvenate

        mu, sigma = prior.sample(n_particles, rng)
        self.mu = mu
        self.sigma = sigma
        # Bumped on every rejuvenation.  Downstream caches of quantities that
        # depend on the particle *values* (not their weights) key on this.
        self.generation = 0
        self.log_lik = np.zeros(n_particles)     # accumulated log-likelihood
        self.log_offset = np.zeros(n_particles)  # importance correction, if any
        self._set_weights()
        self.prev_logw = self.logw.copy()

        self.x_hist: list[float] = []
        self.y_hist: list[int] = []
        self.diag = ParticleDiagnostics()
        self.diag.ess = self.ess()
        self.diag.max_weight = float(self.w.max())

    # -- weights ----------------------------------------------------------
    def _set_weights(self):
        raw = self.log_lik + self.log_offset
        finite = np.isfinite(raw)
        if not np.any(finite):
            raise FloatingPointError("all particles have zero posterior weight")
        norm = logsumexp(raw[finite])
        logw = np.full_like(raw, -np.inf)
        logw[finite] = raw[finite] - norm
        self.logw = logw
        self.w = np.exp(logw)

    def ess(self) -> float:
        return float(1.0 / np.sum(self.w**2))

    @property
    def beta(self):
        """GLM parameterisation beta = (-mu/sigma, 1/sigma)."""
        return -self.mu / self.sigma, 1.0 / self.sigma

    # -- sequential update ------------------------------------------------
    def update(self, x: float, y: int):
        """Absorb one observation and, if triggered, rejuvenate."""
        ll = probit_loglik(np.array([x]), np.array([y]), self.mu, self.sigma)[:, 0]
        self.prev_logw = self.logw.copy()
        self.log_lik = self.log_lik + ll
        self._set_weights()
        self.x_hist.append(float(x))
        self.y_hist.append(int(y))

        kl = self.kl_from_previous()
        self.diag.kl = kl
        self.diag.kl_trace.append(kl)
        if self.rejuvenate and kl > self.kl_threshold:
            self._rejuvenate()
        self.diag.ess = self.ess()
        self.diag.ess_trace.append(self.diag.ess)
        self.diag.max_weight = float(self.w.max())
        return kl

    def kl_from_previous(self) -> float:
        """D_KL(W^(k) || W^(k-1)) = sum_j w_j^(k) [log w_j^(k) - log w_j^(k-1)]."""
        ok = np.isfinite(self.logw) & (self.w > 0)
        if not np.any(ok):
            return float("nan")
        diff = self.logw[ok] - self.prev_logw[ok]
        if not np.all(np.isfinite(diff)):
            return float("inf")
        return float(np.sum(self.w[ok] * diff))

    def _rejuvenate(self):
        b0, b1 = self.beta
        beta_med = np.array([
            lower_inverse_cdf(b0, self.w, 0.5),
            lower_inverse_cdf(b1, self.w, 0.5),
        ])
        B = np.column_stack([b0, b1])
        mean = self.w @ B
        cov = (B - mean).T @ ((B - mean) * self.w[:, None])
        cov = 0.5 * (cov + cov.T)
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov = cov + np.eye(2) * (1e-10 + 1e-6 * np.trace(cov) / 2)
            self.diag.covariance_jitter += 1

        new = self.rng.multivariate_normal(beta_med, cov, size=self.n)
        nb0, nb1 = new[:, 0], new[:, 1]
        invalid = nb1 <= 0
        self.diag.n_invalid_particles += int(invalid.sum())

        with np.errstate(divide="ignore", invalid="ignore"):
            new_mu = np.where(invalid, np.nan, -nb0 / nb1)
            new_sigma = np.where(invalid, np.nan, 1.0 / nb1)

        log_pi0 = self.prior.logpdf_beta(nb0, nb1)
        log_pinew = _mvn_logpdf(new, beta_med, cov)

        safe_mu = np.where(invalid, 1.0, new_mu)
        safe_sigma = np.where(invalid, 1.0, new_sigma)
        ll = np.zeros(self.n)
        if self.x_hist:
            ll = probit_loglik(
                np.asarray(self.x_hist), np.asarray(self.y_hist), safe_mu, safe_sigma
            ).sum(axis=-1)

        self.mu = safe_mu
        self.sigma = safe_sigma
        self.log_lik = np.where(invalid, -np.inf, ll)
        self.log_offset = np.where(invalid, -np.inf, log_pi0 - log_pinew)
        self._set_weights()
        # KL is undefined across different particle supports: restart baseline.
        self.prev_logw = self.logw.copy()
        self.generation += 1
        self.diag.n_resamples += 1
        self.diag.resample_steps.append(len(self.x_hist))

    # -- summaries --------------------------------------------------------
    def values(self, quantity: str):
        return derived_values(self.mu, self.sigma, quantity)

    def mean(self, quantity: str) -> float:
        return float(np.dot(self.w, self.values(quantity)))

    def sd(self, quantity: str) -> float:
        v = self.values(quantity)
        m = float(np.dot(self.w, v))
        return float(np.sqrt(max(np.dot(self.w, (v - m) ** 2), 0.0)))

    def median_rotem(self, quantity: str) -> float:
        """Rotem's estimator: the lower-inverse-CDF posterior median."""
        return float(lower_inverse_cdf(self.values(quantity), self.w, 0.5))

    def quantile(self, quantity: str, p):
        return weighted_quantile(self.values(quantity), self.w, p)

    def cdf_at(self, quantity: str, value: float) -> float:
        v = self.values(quantity)
        return float(np.sum(self.w[v < value]))

    def credible_interval(self, quantity: str, level: float = 0.95):
        a = (1.0 - level) / 2.0
        lo, hi = self.quantile(quantity, [a, 1.0 - a])
        return float(lo), float(hi)

    def covered(self, quantity: str, truth: float, level: float = 0.95) -> bool:
        """Central-credible-interval coverage, read off the rank statistic.

        Same convention as `GridPosterior.covered`, so particle and reference
        coverage are directly comparable and any gap between them is a real
        difference in the represented posterior rather than a difference in
        how the interval was formed.
        """
        a = (1.0 - level) / 2.0
        r = self.cdf_at(quantity, truth)
        return bool(a < r < 1.0 - a)

    def snapshot(self) -> dict:
        return {
            "ess": self.ess(),
            "max_weight": float(self.w.max()),
            "n_resamples": self.diag.n_resamples,
            "n_invalid_particles": self.diag.n_invalid_particles,
            "kl_last": self.diag.kl,
        }


def _mvn_logpdf(x, mean, cov):
    d = x.shape[1]
    L = np.linalg.cholesky(cov)
    diff = solve_triangular(L, (x - mean).T, lower=True)
    quad = np.sum(diff**2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (quad + logdet + d * np.log(2 * np.pi))
