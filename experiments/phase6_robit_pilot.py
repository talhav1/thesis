"""Phase 6, rung 1 -- pilot.  Does enlarging the fitted family widen the tail?

The Phase 3/4A defect is **not** that ``nu = infinity`` is wrong.  It is that
``nu = infinity`` is assumed with *certainty*.  Under the probit,
``q_0.99 = mu + sigma z_0.99`` is a fixed affine function of two parameters the
design pins down near the median, so the posterior for ``q_0.99`` inherits the
precision of data collected nowhere near it.  Phase 4A showed the resulting
error is not merely unnoticed but absent from the sample: in 65.5% of adaptive-MI
runs the misspecified and correct curves generated bit-identical data
(`manuscript/phase4_undetectability_note.md` section 3).  A diagnostic therefore
cannot be the remedy; the remedy has to act on the fitted family or on the
design.

This is the first safeguard rung: replace the fitted probit with a **robit**,
``p(x) = T_{1/r}((x - mu)/sigma)``, carrying a free tail parameter ``r = 1/nu``
on ``[0, 0.5]``.  Phase 4A predicts ``r`` will be nearly unidentified at n = 50.
That is the *mechanism*, not a problem: an unidentified tail parameter should
**widen** ``q_0.99`` rather than sharpen it, which is exactly the transfer of
precision the probit performs and the safeguard is meant to cut.

Four things this script keeps separable, because folding any of them together
would make the result unattributable.

**1.  Fitted family and design are crossed, not bundled.**

    fitted family:  probit   |  robit (free r)
    design:         MI(mu, sigma)  |  MI(mu, sigma, r)

The ``(robit fit, existing design)`` cell isolates the pure *widening* effect;
the ``(probit fit, new design)`` cell isolates the *design* effect.  The second
is a question in its own right: r is identified only in the tail, so a criterion
scoring information about r has a reason to go there that the two-parameter
criterion does not have.  Whether it actually does is measured, not assumed.
Both fits are applied to the **same history**, so the family contrast is exactly
paired within a replicate.

**2.  Three r-priors, always all three.**  r is nearly unidentified, so the
prior is doing most of the work and a conclusion that holds under one prior and
not the others is a statement about the prior.  `docs/decisions.md` DEC-14
records the panel as a reporting rule rather than a tuning knob.  Under the
MI(mu, sigma, r) design the r-prior is a property of the *arm*: an analyst who
reports under a prior designs under it too, so the design and the fit use the
same prior and the cell is coherent.  Under MI(mu, sigma) the design does not
involve r at all, so one history serves all three fits.

**3.  Three DGPs, each answering a different question.**

  * ``probit`` -- the **cost** arm.  The truth is the probit, so any extra width
    is the premium paid for the free parameter.  Compared against the Phase 2
    baseline (coverage 0.945-0.968) and against the probit fit on the same
    histories.
  * ``robit_1.0`` -- the truth is (approximately) *nested in the fitted family*.
    This should be repaired.  If it is not, the implementation is wrong.
  * ``tail_1.0`` -- the truth is **outside** the t family: the perturbation is an
    asymmetric upper-tail shift and the t link is symmetric.  Partial repair at
    best is expected, and the shortfall is reported as a **finding**, not tuned
    around.  Family enlargement repairs *anticipated* misspecification only.

**4.  Metrics are threshold-free and scale-free.**  D16/D17/D18 make the
w*-based false-certainty rate non-comparable across cells whose curves have
different scales, and the robit fit changes interval widths by construction, so
a width threshold frozen under the probit would not mean the same thing here.
Coverage is read off the rank statistic (DEC-3), and beside it go
``z = (q_true - mean)/sd``, the miss in interval half-widths, ``P(|z| > 4)``,
and the dangerous-error rate at ``delta = 0.5 sigma_true``.  The false-certainty
rate is computed as a secondary column where a threshold exists, never as the
primary.

**Horizon.**  n in {50, 100, 200} on the key cells, folded in rather than run
separately, testing two untested predictions at once: under the *probit* fit
false certainty should **worsen** with n (bias fixed by the KL projection, the
interval shrinking like n^-1/2), and under the *robit* fit the q_0.99 interval
should **stop shrinking**, because confidence about a region the design never
samples should not be earned.  Nothing in this project has previously gone above
n = 50.

**Pilot status.**  This run chooses the r-axis resolution from its own
refinement evidence and times a cell.  Nothing it produces is confirmatory and
no replicate count for the confirmatory tier is proposed until
`manuscript/phase6_protocol.md` is frozen against the timing recorded here.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import model_check as mc  # noqa: E402
from src.curve_families import curve_true_sd  # noqa: E402
from src.policies import EntropyVectorPolicy, PolicyState  # noqa: E402
from src.posterior_grid import build_reference_posterior  # noqa: E402
from src.posterior_grid_robit import build_robit_reference_posterior  # noqa: E402
from src.priors import rotem_prior, rotem_stimulus_grid  # noqa: E402
from src.response_models import ProbitCurve  # noqa: E402
from src.robit import R_PRIOR_PANEL, R_PRIORS  # noqa: E402
from src.robit_design import RobitParticlePosterior, RobitPolicyState  # noqa: E402
from src.rotem_particles import ParticlePosterior  # noqa: E402
from src.simulator import build_curve, latent_uniforms, true_targets  # noqa: E402

EXPERIMENT = "phase6_robit_pilot"

# ---------------------------------------------------------------------------
# Fixed settings.  Module level and free of any CLI state: on Windows the only
# start method is `spawn`, which re-imports this module in every child, so
# anything parsed from argv at import time would be re-parsed there.  The CLI is
# read inside `main` and everything a worker needs travels in its task tuple.
# ---------------------------------------------------------------------------

# Phase 3B Block I settings, so a Phase 6 number lands beside a Phase 3 number
# rather than beside a new one.
CALIBRATION = "well"
REF_MU, REF_SIGMA = 30.0, 3.0
GRID_POINTS = 200
GRID_HALF_WIDTH = 14.0
TARGETS = ("q0.5", "q0.95", "q0.99")
LEVEL = 0.95

# Reference resolutions.  The probit side is the DEC-7 production setting,
# unchanged, so the probit-fit column of this experiment is the same estimator
# Phase 3 used.  The robit side is chosen by `resolution_audit` below, which
# runs first and whose evidence is recorded in the manifest's tolerances.
#
# n_r = 33 is the coarsest rung of the ladder whose *estimated* discretisation
# error clears the DEC-7 tolerance of 1e-3 posterior standard deviations.  The
# r axis is second-order (`posterior_grid_robit._r_quadrature_logweights`), so
# the ladder's deltas fall by ~4x per doubling and the Richardson estimate of
# the error remaining at a rung is its gap to the next rung, over 3.  n_r = 17
# does not clear it (about 8e-3 sd), which is why the constant is not 17.
REF_FIXED_N = 513
ROBIT_FIXED_N = 257
ROBIT_N_R = 33

# Particle counts.  **Not** a design factor, and deliberately not shared.
#
# `mi2` runs Rotem's cloud at her own default, N = 10,000 with rejuvenation, so
# the "existing design" arm is her design and not a re-tuned version of it.
# `mi3` carries three parameters and no rejuvenation (`src/robit_design.py`
# explains why her Gaussian refresh in the GLM parameterisation has no meaning
# for r), so its effective sample size decays monotonically and it is given a
# larger cloud.  Both are computational settings chosen to clear the ESS floor
# below, never to move a design number; the design cost is 1-3 s either way,
# two orders below the grid fits, so nothing here trades accuracy for time.
N_PARTICLES = {"mi2": 10_000, "mi3": 40_000}

# Pre-registered adequacy floor on the minimum effective sample size over a
# run.  A cell whose ESS falls below it is reported as computationally
# inadequate rather than as a design finding.  C8 is the precedent: turning
# rejuvenation off beat Rotem's rule despite a 30x lower ESS, so a low ESS is a
# number to report, not a reason to discard a run.  Nothing is ever dropped for
# failing it.
ESS_FLOOR = 100.0

# delta for the dangerous-error rate, as a fraction of the *true* curve's scale
# so the event is comparable across DGPs (phase3 protocol section 4).
DANGEROUS_DELTA_FRAC = 0.5

DGPS = {
    "probit": {"family": "probit", "mu": REF_MU, "sigma": REF_SIGMA},
    "robit_1.0": {"family": "robit", "lam": 1.0},
    "tail_1.0": {"family": "tail_perturbed", "shift95": 1.0},
}

# The 2x2 at the Phase 3 horizon, plus the horizon axis on the key cells only.
# `mi2` is Rotem's design unchanged; `mi3` is the same EntropyVectorPolicy over a
# three-parameter particle cloud -- the policy code is identical, only the
# parameter vector the mutual information is taken over differs.
BASE_N = 50
HORIZONS = (50, 100, 200)
# Key cells for the horizon prediction: the failure cell and its cost control,
# both under the *existing* design, because the prediction is about what happens
# to a probit-fit interval as n grows and what the robit fit does instead.
HORIZON_DGPS = ("probit", "tail_1.0")
HORIZON_DESIGN = "mi2"


def cells() -> list[tuple]:
    """(dgp, design, design_r_prior, n).  `design_r_prior` is '' for mi2."""
    out = []
    for dgp in DGPS:
        out.append((dgp, "mi2", "", BASE_N))
        for rp in R_PRIOR_PANEL:
            out.append((dgp, "mi3", rp, BASE_N))
    for dgp in HORIZON_DGPS:
        for n in HORIZONS:
            if n != BASE_N:
                out.append((dgp, HORIZON_DESIGN, "", n))
    return out


# ---------------------------------------------------------------------------
# One replicate
# ---------------------------------------------------------------------------


def run_design(dgp: str, design: str, design_r_prior: str, n: int,
               replicate: int, seed_base: int, crn_seed: int):
    """Run the sequential design and return (x, y, u, diagnostics).

    Both arms use `EntropyVectorPolicy` unmodified.  The arms differ only in the
    particle cloud handed to it: two-parameter (Rotem's, with her rejuvenation)
    or three-parameter (no rejuvenation -- her Gaussian refresh in the GLM
    parameterisation has no meaning for r, and inventing one would stop being
    her method; see `src/robit_design.py`).  ESS is recorded either way.
    """
    rng = np.random.default_rng([seed_base, replicate])
    prior = rotem_prior(CALIBRATION, "independent")
    curve = build_curve(DGPS[dgp])
    candidates = rotem_stimulus_grid(CALIBRATION, n_points=GRID_POINTS,
                                     half_width=GRID_HALF_WIDTH)
    u = latent_uniforms(crn_seed, replicate, n)

    if design == "mi2":
        cloud = ParticlePosterior(prior, N_PARTICLES["mi2"], rng, rejuvenate=True)
        state = PolicyState(candidates=candidates, posterior=cloud)
    elif design == "mi3":
        cloud = RobitParticlePosterior(prior, R_PRIORS[design_r_prior],
                                       N_PARTICLES["mi3"], rng,
                                       n_r_nodes=ROBIT_N_R)
        state = RobitPolicyState(candidates=candidates, posterior=cloud)
    else:
        raise ValueError(f"unknown design {design!r}")

    policy = EntropyVectorPolicy()
    xs, ys, ess = [], [], []
    for k in range(n):
        x, _info = policy.select(state, rng)
        p_true = float(np.atleast_1d(curve.prob(np.array([x])))[0])
        y = int(u[k] < p_true)
        cloud.update(x, y)
        xs.append(float(x))
        ys.append(y)
        ess.append(float(cloud.ess()))
        state.x_hist = xs
        state.y_hist = ys
        state.step = k + 1
        state.invalidate()

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=int)

    # Where the tail-perturbed curve begins to depart from the matched probit.
    # Defined for every DGP so the cells stay comparable (Phase 4A convention).
    onset = REF_MU + REF_SIGMA * float(mc.norm.ppf(mc.TAIL_P0))
    diag = {
        "x_min": float(x.min()), "x_max": float(x.max()),
        "x_mean": float(x.mean()),
        "n_distinct_x": int(len(np.unique(x))),
        "n_positive": int(y.sum()),
        "all_zero": bool(y.sum() == 0), "all_one": bool(y.sum() == len(y)),
        "perturbation_onset": onset,
        "frac_above_onset": float(np.mean(x > onset)),
        "ess_min": float(np.min(ess)), "ess_final": float(ess[-1]),
        "ess_below_floor": bool(np.min(ess) < ESS_FLOOR),
    }
    # L0 of Phase 4A: what the realised design could reveal at all, and the
    # counterfactual `realized_flips` the blind/informed split is taken on.
    diag.update(mc.design_information(curve, ProbitCurve(REF_MU, REF_SIGMA), x, u))
    return x, y, diag, prior, curve


def _fit_rows(post, truths, sigma_true, fit: str, fit_r_prior: str) -> list[dict]:
    """Per-target metrics from one fitted posterior.

    Coverage is read off the rank statistic (DEC-3) for both fits, so a gap
    between the probit and robit columns is a difference in the posterior and
    never in the convention.  Everything beside it is threshold-free:

    * ``z`` -- the miss in posterior standard deviations, signed so that a
      negative z is an *under*-estimate of the true quantile, the dangerous
      direction for a safety threshold.
    * ``miss_halfwidths`` -- the same miss in units of the reported interval's
      own half-width, which is what a reader of the interval would experience.
    * ``width`` -- primary here, because the horizon prediction is a statement
      about widths and not about coverage.
    """
    rows = []
    for t in TARGETS:
        truth = truths[t]
        m, s = post.mean(t), post.sd(t)
        lo, hi = post.credible_interval(t, LEVEL)
        half = 0.5 * (hi - lo)
        rank = post.cdf_at(t, truth)
        a = (1.0 - LEVEL) / 2.0
        rows.append({
            "fit": fit, "fit_r_prior": fit_r_prior, "target": t,
            "target_true": float(truth),
            "post_mean": float(m), "post_sd": float(s),
            "ci_lo": lo, "ci_hi": hi, "width": float(hi - lo),
            "rank": float(rank), "covered": bool(a < rank < 1.0 - a),
            "err": float(m - truth),
            "z": float((truth - m) / s) if s > 0 else np.nan,
            "miss_halfwidths": float((m - truth) / half) if half > 0 else np.nan,
            "dangerous": bool((m - truth) < -DANGEROUS_DELTA_FRAC * sigma_true),
            "curve_scale": float(sigma_true),
        })
    return rows


def analyse(task) -> tuple[list[dict], list[dict]]:
    """One replicate of one cell: run the design once, then fit both families."""
    import traceback

    (dgp, design, design_r_prior, n, replicate, seed_base, crn_seed,
     robit_fixed_n, robit_n_r) = task
    try:
        t0 = time.time()
        x, y, diag, prior, curve = run_design(dgp, design, design_r_prior, n,
                                              replicate, seed_base, crn_seed)
        t_design = time.time() - t0
        truths = true_targets(curve, TARGETS)
        sigma_true = float(getattr(curve, "sigma", None)
                           or getattr(curve, "implied_sd", None)
                           or curve_true_sd(curve))

        rows = []

        # -- the probit fit: Rotem's reported posterior, unchanged ------------
        t1 = time.time()
        ref = build_reference_posterior(prior, x, y, fixed_n=REF_FIXED_N)
        pr = _fit_rows(ref, truths, sigma_true, "probit", "")
        t_probit = time.time() - t1
        for r in pr:
            r.update({"boundary_mass": float(ref.convergence["boundary_mass"]),
                      "r_mean": np.nan, "r_sd": np.nan,
                      "r_mass_at_zero": np.nan, "r_mass_at_max": np.nan,
                      "fit_seconds": t_probit})
        rows.extend(pr)

        # -- the robit fit ----------------------------------------------------
        # Under mi2 the design carries no r, so one history serves all three
        # priors.  Under mi3 the arm is coherent: the prior that steered the
        # design is the prior the posterior is reported under.
        panel = R_PRIOR_PANEL if design == "mi2" else (design_r_prior,)
        for name in panel:
            t2 = time.time()
            post = build_robit_reference_posterior(
                prior, R_PRIORS[name], x, y,
                fixed_n=robit_fixed_n, fixed_n_r=robit_n_r)
            rr = _fit_rows(post, truths, sigma_true, "robit", name)
            t_robit = time.time() - t2
            rs = post.r_summary()
            for r in rr:
                r.update({"boundary_mass": float(post.convergence["boundary_mass"]),
                          "r_mean": rs["r_mean"], "r_sd": rs["r_sd"],
                          "r_mass_at_zero": rs["r_mass_at_zero"],
                          "r_mass_at_max": rs["r_mass_at_max"],
                          "fit_seconds": t_robit})
            rows.extend(rr)

        head = {"dgp": dgp, "design": design, "design_r_prior": design_r_prior,
                "n": n, "replicate": replicate, "design_seconds": t_design,
                **diag}
        for r in rows:
            r.update(head)
        return rows, []
    except Exception as exc:  # noqa: BLE001 - recorded, never hidden
        return [], [{"dgp": dgp, "design": design,
                     "design_r_prior": design_r_prior, "n": n,
                     "replicate": replicate,
                     "error": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc(limit=6)}]


# ---------------------------------------------------------------------------
# r-axis resolution audit -- item 9 of the pilot's remit
# ---------------------------------------------------------------------------

RESOLUTION_LADDER = (5, 9, 17, 33, 65)


def resolution_audit(seed_base: int, crn_seed: int, n_histories: int = 3) -> pd.DataFrame:
    """Choose the r-axis resolution from evidence rather than asserting it.

    Builds the same posterior on the same histories across an r-ladder and
    records how far each rung sits from the finest, in units of the finest
    rung's own posterior standard deviation.  The production resolution is the
    coarsest rung whose *estimated* error clears the DEC-7 tolerance of 1e-3 sd
    on every tracked target, under every prior in the panel.

    Two columns, and the distinction matters.  ``max_delta_sd`` is the distance
    to the finest rung, which is identically zero *at* the finest rung and so
    says nothing there.  ``rich_err_sd`` is the Richardson estimate of a rung's
    own remaining error: for a second-order scheme the error falls by 4 per
    doubling, so the gap to the next rung is 3/4 of it and the estimate is that
    gap over 3.  It is defined at every rung but the last, which is what makes
    the ladder self-certifying rather than self-referential.

    Run on histories generated by the *adaptive* design, because that is the
    stimulus configuration the experiment actually produces; a resolution
    validated on a spread-out history would not certify it.
    """
    prior = rotem_prior(CALIBRATION, "independent")
    out = []
    for rep in range(n_histories):
        x, y, _diag, _p, _c = run_design("tail_1.0", "mi2", "", BASE_N, rep,
                                         seed_base, crn_seed)
        for name in R_PRIOR_PANEL:
            built = {}
            for n_r in RESOLUTION_LADDER:
                t0 = time.time()
                post = build_robit_reference_posterior(
                    prior, R_PRIORS[name], x, y,
                    fixed_n=ROBIT_FIXED_N, fixed_n_r=n_r)
                built[n_r] = (post, time.time() - t0)
            fine, _ = built[RESOLUTION_LADDER[-1]]
            for i, n_r in enumerate(RESOLUTION_LADDER):
                post, secs = built[n_r]
                rec = {"replicate": rep, "r_prior": name, "n_r": n_r,
                       "build_seconds": secs,
                       "boundary_mass": post.boundary_mass()}
                worst = 0.0
                for t in TARGETS:
                    scale = max(fine.sd(t), 1e-12)
                    dm = abs(post.mean(t) - fine.mean(t)) / scale
                    ds = abs(post.sd(t) - fine.sd(t)) / scale
                    rec[f"{t}_dmean_sd"] = dm
                    rec[f"{t}_dsd_sd"] = ds
                    worst = max(worst, dm, ds)
                rec["max_delta_sd"] = worst
                # Richardson: the gap to the next rung, over 3, estimates the
                # error still present at this one.  Undefined at the finest
                # rung, where there is no next rung to difference against.
                if i + 1 < len(RESOLUTION_LADDER):
                    nxt, _ = built[RESOLUTION_LADDER[i + 1]]
                    gap = 0.0
                    for t in TARGETS:
                        scale = max(nxt.sd(t), 1e-12)
                        gap = max(gap,
                                  abs(post.mean(t) - nxt.mean(t)) / scale,
                                  abs(post.sd(t) - nxt.sd(t)) / scale)
                    rec["rich_err_sd"] = gap / 3.0
                else:
                    rec["rich_err_sd"] = np.nan
                out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

CELL_KEYS = ["dgp", "design", "design_r_prior", "n", "fit", "fit_r_prior", "target"]


def _p(k: int, n: int) -> tuple[float, float]:
    p = k / n if n else np.nan
    se = float(np.sqrt(max(p * (1 - p), 0.0) / n)) if n else np.nan
    return float(p), se


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell metrics.  Every rate carries its Monte Carlo standard error."""
    out = []
    for keys, g in df.groupby(CELL_KEYS, dropna=False):
        n_rep = len(g)
        cov, cov_se = _p(int(g.covered.sum()), n_rep)
        dang, dang_se = _p(int(g.dangerous.sum()), n_rep)
        z = g.z.to_numpy(dtype=float)
        extreme, extreme_se = _p(int(np.sum(np.abs(z) > 4.0)), n_rep)
        rec = dict(zip(CELL_KEYS, keys))
        rec.update({
            "n_rep": n_rep,
            "coverage": cov, "coverage_se": cov_se,
            "z_mean": float(np.nanmean(z)),
            "z_mean_se": float(np.nanstd(z, ddof=1) / np.sqrt(n_rep))
            if n_rep > 1 else np.nan,
            "z_sd": float(np.nanstd(z, ddof=1)) if n_rep > 1 else np.nan,
            "p_abs_z_gt_4": extreme, "p_abs_z_gt_4_se": extreme_se,
            "miss_halfwidths_mean": float(np.nanmean(g.miss_halfwidths)),
            "abs_miss_halfwidths_mean": float(np.nanmean(np.abs(g.miss_halfwidths))),
            "dangerous_error": dang, "dangerous_error_se": dang_se,
            "bias": float(g.err.mean()),
            "bias_se": float(g.err.std(ddof=1) / np.sqrt(n_rep)) if n_rep > 1 else np.nan,
            "rmse": float(np.sqrt((g.err**2).mean())),
            "mean_width": float(g.width.mean()),
            "median_width": float(g.width.median()),
            "mean_post_sd": float(g.post_sd.mean()),
            "target_true": float(g.target_true.iloc[0]),
            "r_mean": float(g.r_mean.mean()),
            "r_sd": float(g.r_sd.mean()),
            "r_mass_at_zero": float(g.r_mass_at_zero.mean()),
            "r_mass_at_max": float(g.r_mass_at_max.mean()),
            "max_boundary_mass": float(g.boundary_mass.max()),
            "ess_min": float(g.ess_min.min()),
            "frac_below_ess_floor": float(g.ess_below_floor.mean()),
            "x_max_mean": float(g.x_max.mean()),
            "frac_above_onset": float(g.frac_above_onset.mean()),
            "design_kl_nats": float(g.design_kl_nats.mean()),
            "realized_flips_mean": float(g.realized_flips.mean()),
            "frac_zero_flips": float((g.realized_flips == 0).mean()),
            "frac_all_equal": float((g.all_zero | g.all_one).mean()),
            "fit_seconds_mean": float(g.fit_seconds.mean()),
            "design_seconds_mean": float(g.design_seconds.mean()),
        })
        # Phase 4A blind/informed split.  `realized_flips == 0` marks a run whose
        # responses are bit-identical to what the correct probit would have
        # produced, so the design collected no signal about the misspecification
        # at all.  The mechanism prediction under test is that the robit fit
        # moves the BLIND runs (probit-fit coverage 0.206 under adaptive MI) and
        # barely touches the INFORMED ones (0.841).  If it moves both equally the
        # stated mechanism is wrong and the write-up has to say so.
        #
        # A counterfactual, therefore a decomposition and not a diagnostic; and
        # the conditioning is post hoc, so it locates the failure rather than
        # estimating the effect of an intervention.
        for label, sub in (("blind", g[g.realized_flips == 0]),
                           ("informed", g[g.realized_flips > 0])):
            k = len(sub)
            c, c_se = _p(int(sub.covered.sum()), k)
            rec[f"n_{label}"] = k
            rec[f"coverage_{label}"] = c
            rec[f"coverage_{label}_se"] = c_se
            rec[f"bias_{label}"] = float(sub.err.mean()) if k else np.nan
            rec[f"mean_width_{label}"] = float(sub.width.mean()) if k else np.nan
        out.append(rec)
    return pd.DataFrame(out).sort_values(CELL_KEYS)


PAIR_KEYS = ["dgp", "design", "design_r_prior", "n", "target"]


def premium(df: pd.DataFrame) -> pd.DataFrame:
    """The width premium, paired within replicate on the identical history.

    The probit and robit fits see the *same* x and y, so the ratio is formed per
    replicate and then averaged -- not as a ratio of two cell means, which would
    confound the family contrast with between-replicate variation in how much
    the design happened to learn.

    Under the `probit` DGP this is the **cost** of the safeguard, and the
    protocol's maximum acceptable premium is stated against it.  Under a
    misspecified DGP it is what the widening bought.
    """
    base = (df[df.fit == "probit"]
            .set_index(PAIR_KEYS + ["replicate"])[["width", "post_sd", "covered"]])
    rob = df[df.fit == "robit"]
    out = []
    for (rp, *keys), g in rob.groupby(["fit_r_prior"] + PAIR_KEYS, dropna=False):
        j = g.set_index(PAIR_KEYS + ["replicate"]).join(
            base, how="inner", rsuffix="_probit")
        if not len(j):
            continue
        ratio = (j.width / j.width_probit).to_numpy(dtype=float)
        ratio = ratio[np.isfinite(ratio)]
        d_cov = (j.covered.astype(float) - j.covered_probit.astype(float)).to_numpy()
        out.append({
            "fit_r_prior": rp, **dict(zip(PAIR_KEYS, keys)), "n_pairs": int(len(j)),
            "width_ratio_mean": float(ratio.mean()) if len(ratio) else np.nan,
            "width_ratio_median": float(np.median(ratio)) if len(ratio) else np.nan,
            "width_ratio_se": float(ratio.std(ddof=1) / np.sqrt(len(ratio)))
            if len(ratio) > 1 else np.nan,
            "width_probit_mean": float(j.width_probit.mean()),
            "width_robit_mean": float(j.width.mean()),
            "coverage_gain": float(d_cov.mean()),
            "coverage_gain_se": float(d_cov.std(ddof=1) / np.sqrt(len(d_cov)))
            if len(d_cov) > 1 else np.nan,
        })
    return pd.DataFrame(out).sort_values(["dgp", "design", "target", "fit_r_prior"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    from _cli import parse as _parse_cli
    from _runner import N_WORKERS, degeneracy_counts, save, save_csv, write_manifest

    run = _parse_cli(EXPERIMENT, replicates=6, seed_base=20260960, crn_seed=616001)
    seed_base, crn_seed, n_reps = run.seed_base, run.crn_seed, run.replicates
    t0 = time.time()

    # The resolution audit runs first: the production r-axis resolution is a
    # constant of this script, and this is what certifies it.  Reporting it
    # after the cells would be publishing a justification for a choice already
    # spent.
    print("--- r-axis resolution audit (certifies ROBIT_N_R) ---", flush=True)
    res = resolution_audit(seed_base, crn_seed)
    rpath = save_csv(res, "phase6_robit_pilot_resolution")
    print(res.groupby(["r_prior", "n_r"])[
        ["max_delta_sd", "rich_err_sd", "build_seconds"]]
        .max().to_string(float_format=lambda v: f"{v:12.6f}"), flush=True)
    chosen = res[res.n_r == ROBIT_N_R]
    print(f"  production n_r = {ROBIT_N_R}: worst distance to the finest rung "
          f"{chosen.max_delta_sd.max():.2e} sd, Richardson error estimate "
          f"{chosen.rich_err_sd.max():.2e} sd", flush=True)

    cs = cells()
    print(f"Phase 6 robit pilot: {len(cs)} cells x {n_reps} replicates "
          f"(targets {TARGETS}, robit grid {ROBIT_FIXED_N}^2 x n_r={ROBIT_N_R})",
          flush=True)
    print("  PILOT TIER. Nothing here is confirmatory.", flush=True)

    tasks = [(dgp, des, rp, n, rep, seed_base, crn_seed, ROBIT_FIXED_N, ROBIT_N_R)
             for (dgp, des, rp, n) in cs for rep in range(n_reps)]

    all_rows, all_fail, done = [], [], 0
    if N_WORKERS == 1:
        results = (analyse(t) for t in tasks)
        pool = None
    else:
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        pool = ctx.Pool(N_WORKERS)
        results = pool.imap_unordered(analyse, tasks)
    for rows, failures in results:
        all_rows.extend(rows)
        all_fail.extend(failures)
        done += 1
        if done % 5 == 0 or done == len(tasks):
            el = time.time() - t0
            print(f"  {done}/{len(tasks)} tasks  {el:7.1f}s "
                  f"(eta {el / done * (len(tasks) - done):7.1f}s)", flush=True)
    if pool is not None:
        pool.close()
        pool.join()

    df = pd.DataFrame(all_rows).sort_values(CELL_KEYS + ["replicate"])
    raw = save(df, "phase6_robit_pilot_raw")
    summ = summarise(df)
    prem = premium(df)
    spath = save_csv(summ, "phase6_robit_pilot_summary")
    ppath = save_csv(prem, "phase6_robit_pilot_premium")

    show = ["dgp", "design", "design_r_prior", "n", "fit", "fit_r_prior",
            "coverage", "coverage_se", "mean_width", "z_mean", "p_abs_z_gt_4",
            "dangerous_error", "r_mean", "coverage_blind", "coverage_informed"]
    for t in TARGETS:
        print(f"\n=== {t} ===")
        print(summ[summ.target == t][show].to_string(
            index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\n=== paired width premium (robit / probit, same history) ===")
    print(prem.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    write_manifest(
        experiment=EXPERIMENT,
        config={"dgps": list(DGPS), "cells": [list(c) for c in cs],
                "targets": list(TARGETS), "horizons": list(HORIZONS),
                "horizon_dgps": list(HORIZON_DGPS),
                "horizon_design": HORIZON_DESIGN,
                "r_prior_panel": list(R_PRIOR_PANEL),
                "r_prior_describe": {k: R_PRIORS[k].describe()
                                     for k in R_PRIOR_PANEL},
                "n_particles": dict(N_PARTICLES), "ess_floor": ESS_FLOOR,
                "credible_level": LEVEL,
                "dangerous_delta_frac": DANGEROUS_DELTA_FRAC,
                "prior": f"rotem {CALIBRATION}/independent",
                "grid_points": GRID_POINTS, "crn_seed": crn_seed,
                "rejuvenate_mi2": True, "rejuvenate_mi3": False,
                "tier": "pilot"},
        seed_base=seed_base, requested=n_reps * len(cs),
        completed=int(len(df.groupby(
            ["dgp", "design", "design_r_prior", "n", "replicate"]))) if len(df) else 0,
        failures=all_fail,
        tolerances={"reference_fixed_n": REF_FIXED_N,
                    "robit_fixed_n": ROBIT_FIXED_N, "robit_n_r": ROBIT_N_R,
                    "robit_resolution_ladder": list(RESOLUTION_LADDER),
                    "robit_resolution_tol_sd": 1e-3,
                    "robit_max_delta_sd_at_production_n_r": float(
                        res[res.n_r == ROBIT_N_R].max_delta_sd.max()),
                    "robit_richardson_err_sd_at_production_n_r": float(
                        res[res.n_r == ROBIT_N_R].rich_err_sd.max()),
                    "max_boundary_mass": float(df.boundary_mass.max()),
                    "min_ess": float(df.ess_min.min()),
                    "ess_floor": ESS_FLOOR,
                    "n_runs_below_ess_floor": int(df.groupby(
                        ["dgp", "design", "design_r_prior", "n", "replicate"])
                        .ess_below_floor.first().sum())},
        mc_se={f"{r.dgp}|{r.design}|{r.design_r_prior}|n{r.n}|{r.fit}"
               f"|{r.fit_r_prior}|{r.target}::coverage": r.coverage_se
               for r in summ.itertuples()},
        degeneracies=degeneracy_counts(df),
        artifacts=[raw, spath, ppath, rpath], wall=time.time() - t0,
        replicates_unit="replicates",
        notes="Phase 6 rung 1 PILOT. Robit fitted family with a free tail "
              "parameter, crossed with an MI design over (mu, sigma, r). "
              "Pilot tier: chooses the r-axis resolution and times a cell. "
              "No number here is confirmatory and no replicate count is "
              "proposed from it before manuscript/phase6_protocol.md is frozen.",
    )


if __name__ == "__main__":
    main()
