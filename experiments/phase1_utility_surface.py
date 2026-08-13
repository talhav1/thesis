"""Phase 1 task 5: the quantile-indexed utility surface U_p(r; H_n).

    U_p(r; H_n) = I( q_p ; Y_{n+1} | X_{n+1} = tilde q_r(H_n), H_n )

Rows index the *target* quantile p; columns index the response level r at which
the next stimulus is placed, with tilde q_r the posterior median of the stimulus
achieving response probability r.

This is the reading of "information gain for all quantiles" that is *not*
redundant.  Proposition 2 shows that targeting more than two quantiles adds
nothing; indexing candidate *stimuli* by response level is a different object
and is exactly the question of where on the sensitivity curve to push.

Evaluated at three posterior states along a single seeded adaptive run -- early
(n=5), intermediate (n=15) and late (n=30) -- so the drift of the surface with
accumulating information is visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _runner import save_csv, write_manifest  # noqa: E402

from src.policies import EntropyVectorPolicy, PolicyState  # noqa: E402
from src.priors import rotem_prior, rotem_stimulus_grid  # noqa: E402
from src.response_models import ProbitCurve  # noqa: E402
from src.rotem_particles import ParticlePosterior  # noqa: E402
from src.simulator import latent_uniforms  # noqa: E402
from src.utilities import utility_surface  # noqa: E402

SEED = 20260813
SNAPSHOTS = (5, 15, 30)
TARGET_PS = (0.5, 0.9, 0.95, 0.99)
STIMULUS_RS = tuple(np.round(np.linspace(0.02, 0.98, 25), 4))


def main():
    prior = rotem_prior("well", "independent")
    curve = ProbitCurve(30.0, 3.0)
    candidates = rotem_stimulus_grid("well", 200, 14.0)

    rng = np.random.default_rng([SEED, 0])
    posterior = ParticlePosterior(prior, 10_000, rng)
    state = PolicyState(candidates=candidates, posterior=posterior)
    policy = EntropyVectorPolicy()
    u = latent_uniforms(SEED, 0, max(SNAPSHOTS))

    records, xs, ys = [], [], []
    for k in range(max(SNAPSHOTS)):
        x, _ = policy.select(state, rng)
        y = int(u[k] < float(np.atleast_1d(curve.prob(np.array([x])))[0]))
        posterior.update(x, y)
        xs.append(x)
        ys.append(y)
        state.x_hist, state.y_hist, state.step = xs, ys, k + 1
        state.invalidate()

        n_done = k + 1
        if n_done in SNAPSHOTS:
            res = utility_surface(posterior, TARGET_PS, STIMULUS_RS)
            for i, p in enumerate(TARGET_PS):
                best = int(np.argmax(res["surface"][i]))
                for j, r in enumerate(STIMULUS_RS):
                    records.append({
                        "n": n_done,
                        "target_p": p,
                        "stimulus_r": r,
                        "stimulus_x": float(res["x_levels"][j]),
                        "utility": float(res["surface"][i, j]),
                        "vector_mi": float(res["vector_mi"][j]),
                        "is_argmax": j == best,
                        "argmax_r": STIMULUS_RS[best],
                        "quad_density_integral":
                            res["quadrature_diagnostics"][i]["quadrature_density_integral"],
                        "quad_empty_windows":
                            res["quadrature_diagnostics"][i]["quadrature_empty_windows"],
                        "post_mu_median": posterior.median_rotem("mu"),
                        "post_sigma_median": posterior.median_rotem("sigma"),
                        "ess": posterior.ess(),
                    })

    df = pd.DataFrame(records)
    path = save_csv(df, "phase1_utility_surface")

    peaks = (df[df.is_argmax]
             .groupby(["n", "target_p"], as_index=False)
             .agg(argmax_r=("stimulus_r", "first"),
                  argmax_x=("stimulus_x", "first"),
                  utility=("utility", "first")))
    peak_path = save_csv(peaks, "phase1_utility_surface_peaks")

    print(peaks.to_string(index=False))
    qd = df.quad_density_integral
    print(f"\nquadrature density integral: min={qd.min():.4f} max={qd.max():.4f}")

    write_manifest(
        experiment="phase1_utility_surface",
        config={"seed": SEED, "snapshots": list(SNAPSHOTS),
                "target_ps": list(TARGET_PS), "stimulus_rs": list(STIMULUS_RS),
                "policy": "entropy_vector", "n_particles": 10_000,
                "prior": prior.spec.describe(),
                "curve": {"family": "probit", "mu": 30.0, "sigma": 3.0}},
        seed_base=SEED,
        requested=1,
        completed=1,
        failures=[],
        tolerances={"quadrature_R": 100, "quadrature_c_fraction": 0.1},
        mc_se={"note": "single seeded path; no replication -- this is a "
                       "structural diagnostic, not an estimate of a population mean"},
        degeneracies={"quad_density_min": float(qd.min()),
                      "quad_empty_max": int(df.quad_empty_windows.max())},
        artifacts=[path, peak_path],
        wall=0.0,
        notes="U_p(r; H_n) at early/intermediate/late posterior states.",
    )


if __name__ == "__main__":
    main()
