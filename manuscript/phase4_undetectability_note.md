# Phase 4A -- is the Phase 3 failure detectable from the data?

**Status: screening tier, 200 replicates per cell, coverage SE ~0.035, power
SE ~0.015.** **Pinned**: clean tree, tag `phase4a-screening`.
Generated from `results/summaries/phase4_undetectability_summary.csv` and
`results/raw/phase4_undetectability_raw.parquet` by
`experiments/build_phase4_report.py`. Do not edit by hand.

Section 3 is derived from the raw table rather than the summary, because the
summary CSV of this pinned run predates the `signal_split` columns that the
experiment now writes. Both come from the same run and the same function, so
the next clean re-run will carry those columns in the summary as well; nothing
here was recomputed outside the runner.

Run `phase4_undetectability_20260823T065938Z_5c4a33`, commit `89ea4d718931f78a80556da9c4d8cb955df292ad`,
3200 replicates completed,
0 failures, {"all_zero": 0, "all_one": 0, "n_rows": 3200}.

---

## 1. The question

Phase 3 established that a curve agreeing with the probit everywhere the
adaptive design looks, and departing from it only at the estimand, drives
q_0.99 coverage to 0.433 ± 0.029 with the narrowest intervals in the study
(`docs/claims.md` C5). The standing objection is that an analyst would notice:
fit a wider model, run a predictive check, look at the residuals. This
experiment asks whether that is so, on the data each design actually collected.

Four layers, in increasing order of what the analyst is assumed to know, all
calibrated against the correct-probit cell **under the same policy** rather
than against an asymptotic null -- n = 50 under a sequentially chosen design is
not an asymptotic regime.

| layer | assumes | statistic |
|---|---|---|
| L0 | nothing; a property of the design | KL between truth and matched probit at the applied stimuli; the Neyman-Pearson ceiling it implies; counterfactual response flips |
| L1 | the default check | mid-p posterior predictive p-values, omnibus and tail-directed |
| L2 | a good analyst | Bayes factors vs logistic, cloglog, robit -- and the oracle tail family frozen at the true shift |
| L3 | the exact truth | 1-df likelihood ratio against the tail family, shift free |

L3 is an upper bound on detectability: no procedure ignorant of the truth can
beat a test that knows the exact functional form and estimates only its size.

## 2. The headline

`tail_1.0`, q_0.99, by design:

| design | q_0.99 coverage | bias | design KL (nats) | E[flips]/50 | runs with 0 flips | power ceiling | best test |
|---|---|---|---|---|---|---|---|
| adaptive MI (mu, sigma) | 0.425 ± 0.035 | -4.76 | 0.353 | 0.48 | 0.655 | 0.381 | **0.050** |
| adaptive MI q_0.95 | 0.560 ± 0.035 | -3.55 | 1.549 | 1.36 | 0.285 | 0.778 | **0.145** |
| non-adaptive fixed | 0.635 ± 0.034 | -4.24 | 0.006 | 0.11 | 0.875 | 0.105 | **0.065** |
| broad exploratory | 0.705 ± 0.032 | -3.58 | 1.134 | 0.60 | 0.540 | 0.799 | **0.275** |

The cell with the worst coverage in the entire experiment is the cell where the
best of five tests achieves **exactly the nominal level**. Under the adaptive
MI design, q_0.99 coverage is 0.425 ± 0.035 and
the most powerful check available -- including one handed the true curve --
detects the misspecification at 0.050, against a nominal 0.05.

The mechanism is visible in one number. The perturbation begins at x = 33.11;
the adaptive design's mean maximum stimulus is 36.27, and it
spends 0.302 of its budget above the onset. Across
200 replicates the expected number of the 50 responses that would have come
out differently had the truth been the probit is 0.48,
and in 0.655 of runs **not one response differs**. In
those runs the two hypotheses did not go undetected: they generated the same
data, bit for bit. No test can separate them, and none does.

The independent check on this is that q_0.99 coverage here,
0.425 ± 0.035, reproduces the Phase 3B
confirmatory 0.433 ± 0.029 through an entirely separate code path.

## 3. Where the failure lives

Splitting each cell by whether the run collected any discriminating signal at
all -- `realized_flips = 0` means the 50 responses are bit-identical to what
the correct probit would have produced from the same latent draws:

| design | blind runs | coverage | bias | informed runs | coverage | bias |
|---|---|---|---|---|---|---|
| adaptive MI (mu, sigma) | 131 | **0.206 ± 0.035** | -6.11 | 69 | **0.841 ± 0.044** | -2.18 |
| adaptive MI q_0.95 | 57 | **0.123 ± 0.044** | -6.25 | 143 | **0.734 ± 0.037** | -2.47 |
| non-adaptive fixed | 175 | **0.606 ± 0.037** | -4.58 | 25 | **0.840 ± 0.075** | -1.91 |
| broad exploratory | 108 | **0.472 ± 0.048** | -5.63 | 92 | **0.978 ± 0.015** | -1.17 |

**When the design collects even one bit about the misspecification, coverage is
close to nominal. When it collects none, coverage collapses.** Exploratory runs
that land on signal reach 0.978; the same design's blind runs sit at
0.472. Under adaptive MI the gap is 0.206 against
0.841, and the bias roughly triples across it
(-6.11 versus -2.18).

This reframes the mechanism. Broad exploration is not weak; it is
**unreliably aimed**. It reaches near-nominal coverage on the runs where it
happens to land where the curves differ, and only
0.46 of its runs do. That is why it improves coverage
substantially without restoring it, which is the standing negative result C10,
and it says what a Phase 6 safeguard has to achieve: not more exploration but
exploration that reliably lands where the model's implications are least
constrained by what has been collected.

Two cautions. The split variable is a **counterfactual** -- computing it needs
the true curve -- so this is a mechanism decomposition and not a diagnostic.
Its use to Phase 5 is as the target a computable proxy would have to
approximate. And the conditioning is post hoc: these are not randomised arms,
so the contrast identifies where the failure concentrates, not the effect of an
intervention that forces a design to collect signal.

## 4. Detection rates, all cells

Rejection rate at level 0.05, critical values from the correct-probit cell
under the same policy.

| DGP | design | coverage | PPC chi2 | PPC tail | BF guessable | BF oracle | LR oracle |
|---|---|---|---|---|---|---|---|
| probit | adaptive MI (mu, sigma) | 0.970 | 0.065 | 0.050 | 0.050 | 0.050 | 0.050 |
| probit | adaptive MI q_0.95 | 0.970 | 0.050 | 0.055 | 0.050 | 0.050 | 0.050 |
| probit | non-adaptive fixed | 0.950 | 0.050 | 0.055 | 0.050 | 0.050 | 0.050 |
| probit | broad exploratory | 0.925 | 0.050 | 0.050 | 0.050 | 0.050 | 0.050 |
| robit_1.0 | adaptive MI (mu, sigma) | 0.590 | 0.070 | 0.040 | 0.040 | 0.050 | 0.055 |
| robit_1.0 | adaptive MI q_0.95 | 0.575 | 0.050 | 0.050 | 0.105 | 0.080 | 0.055 |
| robit_1.0 | non-adaptive fixed | 0.765 | 0.050 | 0.070 | 0.045 | 0.030 | 0.040 |
| robit_1.0 | broad exploratory | 0.805 | 0.205 | 0.185 | 0.230 | 0.175 | 0.160 |
| tail_0.5 | adaptive MI (mu, sigma) | 0.790 | 0.060 | 0.045 | 0.035 | 0.050 | 0.045 |
| tail_0.5 | adaptive MI q_0.95 | 0.795 | 0.045 | 0.080 | 0.065 | 0.110 | 0.075 |
| tail_0.5 | non-adaptive fixed | 0.850 | 0.050 | 0.065 | 0.045 | 0.050 | 0.045 |
| tail_0.5 | broad exploratory | 0.860 | 0.075 | 0.100 | 0.080 | 0.120 | 0.115 |
| tail_1.0 | adaptive MI (mu, sigma) | 0.425 | 0.050 | 0.040 | 0.030 | 0.045 | 0.040 |
| tail_1.0 | adaptive MI q_0.95 | 0.560 | 0.065 | 0.090 | 0.105 | 0.145 | 0.105 |
| tail_1.0 | non-adaptive fixed | 0.635 | 0.050 | 0.065 | 0.045 | 0.055 | 0.050 |
| tail_1.0 | broad exploratory | 0.705 | 0.160 | 0.200 | 0.175 | 0.275 | 0.230 |

Three readings.

**Exploration buys detectability and coverage together.** For `tail_1.0`,
moving from the adaptive MI design to the broad exploratory one raises coverage
from 0.425 to 0.705,
the design KL from 0.353 to 1.134 nats, and
oracle Bayes-factor power from 0.045 to 0.275.
Detectability is not a separate objective competing with accuracy; on this
family the two move together.

**Undetectability is a property of the design, not of the family.** The robit
DGP is detectable under the exploratory design (best test
0.230)
and undetectable under the adaptive one
(0.070),
at coverage 0.590.
The same misspecification is caught or missed according to where the budget
went.

**Detectability tracks what the design collected, not how badly inference
failed.** Across the 12 misspecified cells, the correlation between the
design KL and oracle Bayes-factor power is 0.74; between coverage and
that power it is 0.15.
The cell that fails worst (tail_1.0 under adaptive MI (mu, sigma),
coverage 0.425) is the one least likely to be caught.

## 5. Null calibration

Every test is at its nominal level on the correct-probit cell, which is what
makes the power columns above readable as power rather than as miscalibration.

| design | test | critical value | realised level |
|---|---|---|---|
| adaptive MI q_0.95 | lr_oracle | 0.972 | 0.050 ± 0.015 |
| adaptive MI q_0.95 | ppc_chi2 | 0.264 | 0.050 ± 0.015 |
| adaptive MI q_0.95 | ppc_tail | 0.440 | 0.055 ± 0.016 |
| adaptive MI q_0.95 | bf_oracle | 0.662 | 0.050 ± 0.015 |
| adaptive MI q_0.95 | bf_guessable | 0.544 | 0.050 ± 0.015 |
| adaptive MI (mu, sigma) | lr_oracle | 0.209 | 0.050 ± 0.015 |
| adaptive MI (mu, sigma) | ppc_chi2 | 0.315 | 0.065 ± 0.017 |
| adaptive MI (mu, sigma) | ppc_tail | 0.265 | 0.050 ± 0.015 |
| adaptive MI (mu, sigma) | bf_oracle | 0.362 | 0.050 ± 0.015 |
| adaptive MI (mu, sigma) | bf_guessable | 0.529 | 0.050 ± 0.015 |
| non-adaptive fixed | lr_oracle | 2.289 | 0.050 ± 0.015 |
| non-adaptive fixed | ppc_chi2 | 0.140 | 0.050 ± 0.015 |
| non-adaptive fixed | ppc_tail | 0.505 | 0.055 ± 0.016 |
| non-adaptive fixed | bf_oracle | 0.797 | 0.050 ± 0.015 |
| non-adaptive fixed | bf_guessable | 0.831 | 0.050 ± 0.015 |
| broad exploratory | lr_oracle | 2.108 | 0.050 ± 0.015 |
| broad exploratory | ppc_chi2 | 0.305 | 0.050 ± 0.015 |
| broad exploratory | ppc_tail | 0.485 | 0.050 ± 0.015 |
| broad exploratory | bf_oracle | 0.856 | 0.050 ± 0.015 |
| broad exploratory | bf_guessable | 0.916 | 0.050 ± 0.015 |

## 6. What would change these conclusions

- **Screening precision.** Power SE ~0.015 per cell; the critical values carry
  their own Monte Carlo error at 200 null replicates. A confirmatory tier
  should re-estimate both together.
- **Reaching the tail is necessary but not sufficient.** A design placed
  *entirely* above the onset has no power either: with nothing near the median
  the probit re-fits its location and scale and absorbs the perturbation
  (`tests/test_model_check.py::test_reaching_the_tail_alone_does_not_confer_power`).
  What a safeguard must preserve is the *contrast* between the region that pins
  the location-scale and the region carrying the estimand -- a sharper
  prescription than "explore more", and one this experiment pins down.
- **The oracle is generous and still fails.** L2's oracle alternative is the
  exact shape of the truth with its magnitude frozen at the true value, and L3
  knows the shape and estimates the magnitude. Neither is available to a real
  analyst. That the failure survives both is what makes the negative result
  strong rather than a comment on weak diagnostics.
- **One curve family, one horizon.** As with C5, the tail-perturbed family is
  adversarial by construction. Whether real sensitivity curves behave this way
  is not a statistical question, and the thesis should keep saying so.
- **Numerics.** Evidence integrals on a 385-point shared
  box; maximum boundary mass over all 3200 rows
  1.4e-05, reference posterior boundary mass
  4.6e-10. Bayes factors are stable to ~1e-4 under
  grid refinement and box enlargement
  (`tests/test_model_check.py::test_bayes_factors_are_converged_and_box_invariant`).

## 7. Consequence for the plan

A safeguard is not optional and cannot be a diagnostic. The failure is not
merely unnoticed by the checks an analyst runs -- it is absent from the data
those checks read. Any remedy therefore has to act on the **design** or on the
**reported interval**, not on model criticism after the fact, because there is
nothing in the sample for model criticism to find.
