"""Priors on (mu, sigma) matching Rotem's simulation settings.

Rotem specifies the prior through the experimenter's beliefs about the
sensitivity curve's location and scale:

    mu    ~ N(mu0, sigma_mu^2)          truncated to mu > 0
    alpha ~ LogNormal(log 0.08, tau^2)  ("sd is proportional to the mean")

and then considers two variants:

  * **dependent**  : sigma = alpha * mu
  * **independent**: sigma ~ LogNormal(log(mu0) + log(0.08), tau^2)

Both a sampler and a *density* are provided.  The density is needed twice:
by the reference grid posterior, and by the importance-sampling correction in
Rotem's rejuvenation step, which reweights by pi_0(beta)/pi_new(beta).

Ambiguity, recorded as discrepancy D1: the thesis writes the log-normal
dispersion as "tau^2_alpha = 0.75" but later calls tau_alpha a *scale*
parameter.  `tau_is_variance` selects the reading; the default treats 0.75 as
the variance (tau = 0.866).  Both readings are exercised in the sensitivity
check in `experiments/phase1_prior_sensitivity.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

LOG_ALPHA0 = np.log(0.08)


@dataclass(frozen=True)
class PriorSpec:
    """Configuration of Rotem's (mu, sigma) prior."""

    mu0: float = 30.0
    sigma_mu: float = 6.0
    log_alpha0: float = LOG_ALPHA0
    tau: float = 0.75
    tau_is_variance: bool = True
    dependent: bool = False
    name: str = "prior"

    @property
    def tau_scale(self) -> float:
        return float(np.sqrt(self.tau)) if self.tau_is_variance else float(self.tau)

    @property
    def sigma_log_location(self) -> float:
        """Location of the marginal log-normal for sigma (independent case)."""
        return float(np.log(self.mu0) + self.log_alpha0)

    def config(self) -> dict:
        """Constructor arguments only -- round-trips through `PriorSpec(**d)`."""
        return {
            "mu0": self.mu0,
            "sigma_mu": self.sigma_mu,
            "log_alpha0": self.log_alpha0,
            "tau": self.tau,
            "tau_is_variance": self.tau_is_variance,
            "dependent": self.dependent,
            "name": self.name,
        }

    def describe(self) -> dict:
        """Config plus derived quantities, for manifests and reports."""
        return {**self.config(), "tau_scale": self.tau_scale,
                "sigma_log_location": self.sigma_log_location}


class Prior:
    """Sampler + log-density for (mu, sigma) under a `PriorSpec`."""

    def __init__(self, spec: PriorSpec):
        self.spec = spec

    # -- sampling ---------------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator):
        """Draw n parameter vectors.

        mu is truncated at 0 by oversampling and keeping the first n positive
        draws, exactly as Rotem describes ("sample some extra prior
        observations of mu and take the first N positive ones").
        """
        s = self.spec
        mu = _sample_positive_normal(n, s.mu0, s.sigma_mu, rng)
        alpha = rng.lognormal(mean=s.log_alpha0, sigma=s.tau_scale, size=n)
        if s.dependent:
            sigma = alpha * mu
        else:
            sigma = rng.lognormal(
                mean=s.sigma_log_location, sigma=s.tau_scale, size=n
            )
        return mu, sigma

    # -- density ----------------------------------------------------------
    def logpdf(self, mu, sigma):
        """log pi_0(mu, sigma), normalised up to the mu>0 truncation constant.

        The truncation constant is included so that the density integrates to
        one; this matters for the grid posterior's normalising-constant checks.
        """
        s = self.spec
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        out = np.full(np.broadcast(mu, sigma).shape, -np.inf)
        ok = (mu > 0) & (sigma > 0) & np.isfinite(mu) & np.isfinite(sigma)
        if not np.any(ok):
            return out
        mu_ok = np.broadcast_to(mu, out.shape)[ok]
        sig_ok = np.broadcast_to(sigma, out.shape)[ok]

        log_trunc = np.log(1.0 - norm.cdf(0.0, loc=s.mu0, scale=s.sigma_mu))
        lp_mu = norm.logpdf(mu_ok, loc=s.mu0, scale=s.sigma_mu) - log_trunc

        if s.dependent:
            # sigma = alpha * mu with alpha independent of mu, so
            # p(sigma | mu) = p_alpha(sigma/mu) / mu.
            lp_sigma = (
                _lognormal_logpdf(sig_ok / mu_ok, s.log_alpha0, s.tau_scale)
                - np.log(mu_ok)
            )
        else:
            lp_sigma = _lognormal_logpdf(
                sig_ok, s.sigma_log_location, s.tau_scale
            )
        out[ok] = lp_mu + lp_sigma
        return out

    def logpdf_beta(self, beta0, beta1):
        """log pi_0 in the GLM parameterisation beta = (-mu/sigma, 1/sigma).

        Includes the Jacobian |d(mu,sigma)/d(beta0,beta1)| = 1/|beta1|^3.
        Used by the rejuvenation importance correction, which Rotem carries
        out in the beta parameterisation.
        """
        beta0 = np.asarray(beta0, dtype=float)
        beta1 = np.asarray(beta1, dtype=float)
        out = np.full(np.broadcast(beta0, beta1).shape, -np.inf)
        ok = beta1 > 0
        if not np.any(ok):
            return out
        b0 = np.broadcast_to(beta0, out.shape)[ok]
        b1 = np.broadcast_to(beta1, out.shape)[ok]
        mu = -b0 / b1
        sigma = 1.0 / b1
        out[ok] = self.logpdf(mu, sigma) - 3.0 * np.log(b1)
        return out

    def describe(self):
        return self.spec.describe()


def _sample_positive_normal(n, loc, scale, rng, chunk_factor=1.2):
    """First n positive draws from N(loc, scale^2)."""
    out = np.empty(n)
    filled = 0
    while filled < n:
        m = max(int((n - filled) * chunk_factor) + 16, 64)
        draw = rng.normal(loc, scale, size=m)
        draw = draw[draw > 0]
        take = min(len(draw), n - filled)
        out[filled : filled + take] = draw[:take]
        filled += take
    return out


def _lognormal_logpdf(x, location, scale):
    x = np.asarray(x, dtype=float)
    return -np.log(x) - np.log(scale) - 0.5 * np.log(2 * np.pi) - 0.5 * (
        (np.log(x) - location) / scale
    ) ** 2


# --------------------------------------------------------------------------
# The four named prior settings of Rotem's Table 1
# --------------------------------------------------------------------------

WELL_CALIBRATED_MU0 = 30.0
POORLY_CALIBRATED_MU0 = 22.0


def rotem_prior(calibration: str, dependence: str) -> Prior:
    """Build one of Rotem's four prior settings."""
    if calibration not in {"well", "poor"}:
        raise ValueError(f"calibration must be 'well' or 'poor', got {calibration!r}")
    if dependence not in {"dependent", "independent"}:
        raise ValueError(
            f"dependence must be 'dependent' or 'independent', got {dependence!r}"
        )
    mu0 = WELL_CALIBRATED_MU0 if calibration == "well" else POORLY_CALIBRATED_MU0
    spec = PriorSpec(
        mu0=mu0,
        sigma_mu=6.0,
        dependent=(dependence == "dependent"),
        name=f"{calibration}_{dependence}",
    )
    return Prior(spec)


def rotem_stimulus_grid(calibration: str, n_points: int = 200, half_width: float = 14.0):
    """Rotem's candidate stimulus set: 200 points, +/- 14 units around mu0.

    Note the consequence, flagged in the Phase 0 note: under poor calibration
    the grid is [8, 36], which *excludes* the true x_0.99 = 36.98.  That is a
    design-support violation present in Rotem's own settings, not something
    this project introduced.
    """
    mu0 = WELL_CALIBRATED_MU0 if calibration == "well" else POORLY_CALIBRATED_MU0
    return np.linspace(mu0 - half_width, mu0 + half_width, n_points)
