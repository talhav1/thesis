"""Phase 1 task 6: audit Rotem's scalar mutual-information quadrature.

Against a closed-form reference.  At the prior, with mu ~ N(mu0, s_mu^2)
independent of sigma,

    P(Y = 1 | sigma, x) = E_mu[Phi((x - mu)/sigma)]
                        = Phi( (x - mu0) / sqrt(sigma^2 + s_mu^2) ),

so I(sigma; Y | x) = h(E_sigma[P]) - E_sigma[h(P)] reduces to a one-dimensional
integral over sigma that can be evaluated to machine precision.  The same trick
gives I(mu; Y | x) exactly, since P(Y = 1 | mu, x) = E_sigma[Phi((x-mu)/sigma)].

This is the only place in the project where the *design criterion* itself -- as
opposed to the posterior -- can be checked against an exact answer, and it is
worth doing: section 5.3's estimator is a ratio of two rectangular kernel
density estimates, and nothing in the thesis bounds its error.

Two things are measured:

1. **Bias.** Kernel smoothing of pi(theta | y=1, x) and pi(theta | y=0, x)
   pulls P_{r,x} towards 1/2, which inflates the conditional entropy and so
   *deflates* the estimated mutual information.  The estimate can go negative,
   which no mutual information can be.
2. **Whether the bias changes the design.** A criterion may be biased and still
   select the same stimulus.  The argmax is compared directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _runner import save_csv, write_manifest  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from src.priors import rotem_prior  # noqa: E402
from src.response_models import binary_entropy, derived_values  # noqa: E402
from src.rotem_particles import ParticlePosterior  # noqa: E402
from src.utilities import (  # noqa: E402
    mutual_information_scalar,
    mutual_information_vector,
    response_probability_matrix,
)

SEED = 20260817
N_PARTICLES = 40_000
N_NODES = 20_001
C_FRACTIONS = (0.02, 0.05, 0.1, 0.2, 0.4)
R_VALUES = (50, 100, 200, 400)


def exact_mi(spec, xs, target: str, n_nodes=N_NODES):
    """Closed-form I(theta; Y | x) at the prior, for theta = sigma or mu."""
    u = np.linspace(-8, 8, n_nodes)
    w = np.exp(-0.5 * u**2)
    w /= w.sum()
    if target == "sigma":
        sig = np.exp(spec.sigma_log_location + spec.tau_scale * u)
        P = ndtr((xs[None, :] - spec.mu0) / np.sqrt(sig[:, None] ** 2 + spec.sigma_mu**2))
    elif target == "mu":
        mu = spec.mu0 + spec.sigma_mu * u
        # P(Y=1 | mu, x) = E_sigma[Phi((x-mu)/sigma)], by quadrature over sigma
        v = np.linspace(-8, 8, 2001)
        wv = np.exp(-0.5 * v**2)
        wv /= wv.sum()
        sig = np.exp(spec.sigma_log_location + spec.tau_scale * v)
        P = np.empty((len(mu), len(xs)))
        for i, m in enumerate(mu):
            P[i] = wv @ ndtr((xs[None, :] - m) / sig[:, None])
    else:
        raise ValueError(target)
    return binary_entropy(w @ P) - w @ binary_entropy(P)


def main():
    prior = rotem_prior("well", "independent")
    spec = prior.spec
    xs = np.linspace(spec.mu0 - 14, spec.mu0 + 14, 200)

    rng = np.random.default_rng(SEED)
    post = ParticlePosterior(prior, N_PARTICLES, rng)
    P = response_probability_matrix(xs, post.mu, post.sigma)
    mi_vec = mutual_information_vector(post.w, P)

    rows = []
    for target in ("sigma", "mu"):
        ex = exact_mi(spec, xs, target)
        theta = derived_values(post.mu, post.sigma, target)
        for c in C_FRACTIONS:
            for R in R_VALUES:
                res = mutual_information_scalar(theta, post.w, P, R=R, c_fraction=c)
                err = res.mi - ex
                rows.append({
                    "target": target, "c_fraction": c, "R": R,
                    "exact_max": float(ex.max()),
                    "exact_argmax_x": float(xs[int(np.argmax(ex))]),
                    "est_argmax_x": float(xs[int(np.argmax(res.mi))]),
                    "argmax_matches": bool(np.argmax(ex) == np.argmax(res.mi)),
                    # What acting on the estimate actually costs: the exact
                    # criterion value at the stimulus the estimate selects,
                    # relative to the exact optimum.  Index equality is the
                    # wrong measure when the exact criterion has ties -- and
                    # I(sigma; Y | x) is exactly symmetric about mu0, so its
                    # two grid-edge maxima are tied to 1e-12.
                    "exact_value_at_est_argmax": float(ex[int(np.argmax(res.mi))]),
                    "criterion_efficiency":
                        float(ex[int(np.argmax(res.mi))] / max(ex.max(), 1e-12)),
                    "mean_bias": float(err.mean()),
                    "max_abs_error": float(np.abs(err).max()),
                    "bias_over_exact_max": float(err.mean() / max(ex.max(), 1e-12)),
                    "frac_negative": float((res.mi < 0).mean()),
                    "min_estimate": float(res.mi.min()),
                    "corr_with_exact": float(np.corrcoef(res.mi, ex)[0, 1]),
                    "density_integral": res.density_integral,
                })
    df = pd.DataFrame(rows)
    path = save_csv(df, "phase1_quadrature_audit")

    default = df[(df.c_fraction == 0.1) & (df.R == 100)]
    print("At Rotem's stated settings (c = 0.1 sd, R = 100):")
    print(default.to_string(index=False, float_format=lambda v: f"{v:10.5f}"))
    print("\nBias vs bandwidth (R = 100):")
    print(df[df.R == 100].pivot(index="c_fraction", columns="target",
                                values="mean_bias")
          .to_string(float_format=lambda v: f"{v:10.6f}"))
    print("\nCriterion efficiency -- exact criterion value at the stimulus the "
          "estimate picks, relative to the exact optimum:")
    print(df.pivot_table(index=["target", "c_fraction"], columns="R",
                         values="criterion_efficiency")
          .to_string(float_format=lambda v: f"{v:8.5f}"))

    # magnitude context: how big is the sigma criterion compared to the bias?
    ex_sigma = exact_mi(spec, xs, "sigma")
    ex_mu = exact_mi(spec, xs, "mu")
    print(f"\nexact max I(sigma;Y) = {ex_sigma.max():.5f} nats   "
          f"exact max I(mu;Y) = {ex_mu.max():.5f} nats   "
          f"max I((mu,sigma);Y) = {mi_vec.max():.5f} nats")
    print(f"I(sigma;Y) is exactly 0 at x = mu0 = {spec.mu0} "
          f"(computed {ex_sigma[np.argmin(np.abs(xs - spec.mu0))]:.2e}) and rises "
          f"monotonically to the grid edges -- so the sigma-targeted design is "
          f"driven to whatever bounds the experimenter chose.")

    write_manifest(
        experiment="phase1_quadrature_audit",
        config={"seed": SEED, "n_particles": N_PARTICLES,
                "c_fractions": list(C_FRACTIONS), "R_values": list(R_VALUES),
                "n_reference_nodes": N_NODES, "prior": spec.describe(),
                "stimulus_grid": [float(xs[0]), float(xs[-1]), len(xs)]},
        seed_base=SEED,
        requested=1, completed=1, failures=[],
        tolerances={"reference": "1-D Gauss-Hermite-style quadrature, 20001 nodes"},
        mc_se={"note": "particle Monte Carlo error at N=40000 is ~1e-3 nats; "
                       "the reference is deterministic"},
        degeneracies={"frac_negative_at_default":
                      float(default.frac_negative.max())},
        artifacts=[path],
        wall=0.0,
        notes="Exactness audit of the section 5.3 scalar mutual-information "
              "quadrature against a closed-form prior-predictive reference.",
    )


if __name__ == "__main__":
    main()
