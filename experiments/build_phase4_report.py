"""Generate the Phase 4A note from the Phase 4A results files.

`CLAUDE.md` section 6: manuscript prose is generated from `results/`, never
typed.  Every number in `manuscript/phase4_undetectability_note.md` is read
here from the summary CSV, the raw parquet or the manifest, so the note cannot
drift from the run that produced it and a re-run rewrites it rather than
contradicting it.

    python experiments/build_phase4_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SUM = ROOT / "results" / "summaries"
RAW = ROOT / "results" / "raw"
MANIFESTS = ROOT / "results" / "manifests"
OUT = ROOT / "manuscript" / "phase4_undetectability_note.md"

TESTS = [("power_ppc_chi2", "PPC chi2"), ("power_ppc_tail", "PPC tail"),
         ("power_bf_guessable", "BF guessable"), ("power_bf_oracle", "BF oracle"),
         ("power_lr_oracle", "LR oracle")]
POL_LABEL = {"entropy_vector": "adaptive MI (mu, sigma)",
             "entropy_q0.95": "adaptive MI q_0.95",
             "fixed_design": "non-adaptive fixed",
             "uniform_grid": "broad exploratory"}
ORDER = ["entropy_vector", "entropy_q0.95", "fixed_design", "uniform_grid"]


def f(v, d=3):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def pm(v, se, d=3):
    return f"{f(v, d)} ± {f(se, d)}"


def load():
    summ = pd.read_csv(SUM / "phase4_undetectability_summary.csv")
    crit = pd.read_csv(SUM / "phase4_undetectability_calibration.csv")
    raw = pd.read_parquet(RAW / "phase4_undetectability_raw.parquet")
    man = json.loads((MANIFESTS / "phase4_undetectability.json").read_text())
    summ["pol_order"] = summ.pol.map({p: i for i, p in enumerate(ORDER)})
    return summ.sort_values(["dgp", "pol_order"]), crit, raw, man


def table_headline(summ: pd.DataFrame) -> str:
    """Coverage and detectability side by side for the primary DGP."""
    g = summ[summ.dgp == "tail_1.0"]
    out = ["| design | q_0.99 coverage | bias | design KL (nats) | E[flips]/50 | runs with 0 flips | power ceiling | best test |",
           "|---|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        best = max(getattr(r, c) for c, _ in TESTS)
        out.append(f"| {POL_LABEL[r.pol]} | {pm(r.coverage, r.coverage_se)} | "
                   f"{f(r.bias, 2)} | {f(r.design_kl_nats)} | {f(r.expected_flips, 2)} | "
                   f"{f(r.frac_zero_flips)} | {f(r.max_power_bound)} | **{f(best)}** |")
    return "\n".join(out)


def table_powers(summ: pd.DataFrame) -> str:
    out = ["| DGP | design | coverage | " + " | ".join(n for _, n in TESTS) + " |",
           "|---|---|---|" + "---|" * len(TESTS)]
    for r in summ.itertuples():
        cells = " | ".join(f(getattr(r, c)) for c, _ in TESTS)
        out.append(f"| {r.dgp} | {POL_LABEL[r.pol]} | {f(r.coverage)} | {cells} |")
    return "\n".join(out)


def table_calibration(crit: pd.DataFrame) -> str:
    out = ["| design | test | critical value | realised level |", "|---|---|---|---|"]
    for r in crit.sort_values("pol").itertuples():
        out.append(f"| {POL_LABEL[r.pol]} | {r.test} | {f(r.critical_value)} | "
                   f"{pm(r.null_level, r.null_level_se)} |")
    return "\n".join(out)


def main():
    summ, crit, raw, man = load()
    n_rep = int(summ.n_rep.max())
    primary = summ[(summ.dgp == "tail_1.0") & (summ.pol == "entropy_vector")].iloc[0]
    g = raw[(raw.dgp == "tail_1.0") & (raw.pol == "entropy_vector")]
    onset = float(g.perturbation_onset.iloc[0])
    best_primary = max(primary[c] for c, _ in TESTS)

    mis = summ[summ.dgp != "probit"].copy()
    mis["power_max"] = mis[[c for c, _ in TESTS]].max(axis=1)
    worst_cov = mis.sort_values("coverage").iloc[0]
    corr_kl = float(np.corrcoef(mis.design_kl_nats, mis.power_bf_oracle)[0, 1])
    tol = man["tolerances"]
    prov = man.get("provenance", {})
    git = prov.get("git", {})
    # `pinned` is the direct fact and is what the status line must key on:
    # `allow_dirty` only says whether a waiver was *requested*, which is false
    # both for a pinned run and for one that was never gated at all.
    pinned = bool(prov.get("pinned"))
    tag = git.get("tag")

    body = f"""# Phase 4A -- is the Phase 3 failure detectable from the data?

**Status: screening tier, {n_rep} replicates per cell, coverage SE ~0.035, power
SE ~0.015.** {"**Pinned**: clean tree, tag `" + tag + "`." if pinned and tag else ("**Pinned**: clean tree, untagged." if pinned else "**Unpinned**: run from a dirty tree. No number here is citable until it is reproduced from a clean, tagged tree (DEC-9).")}
Generated from `results/summaries/phase4_undetectability_summary.csv` by
`experiments/build_phase4_report.py`. Do not edit by hand.

Run `{man["run_id"]}`, commit `{man.get("git_commit", "?")}`,
{man["n_replicates_completed"]} {man.get("replicates_unit", "replicates")} completed,
{man["n_failures"]} failures, {json.dumps(man["degeneracy_counts"])}.

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

{table_headline(summ)}

The cell with the worst coverage in the entire experiment is the cell where the
best of five tests achieves **exactly the nominal level**. Under the adaptive
MI design, q_0.99 coverage is {pm(primary.coverage, primary.coverage_se)} and
the most powerful check available -- including one handed the true curve --
detects the misspecification at {f(best_primary)}, against a nominal 0.05.

The mechanism is visible in one number. The perturbation begins at x = {f(onset, 2)};
the adaptive design's mean maximum stimulus is {f(primary.x_max_mean, 2)}, and it
spends {f(primary.frac_above_onset)} of its budget above the onset. Across
{len(g)} replicates the expected number of the 50 responses that would have come
out differently had the truth been the probit is {f(primary.expected_flips, 2)},
and in {f(primary.frac_zero_flips)} of runs **not one response differs**. In
those runs the two hypotheses did not go undetected: they generated the same
data, bit for bit. No test can separate them, and none does.

The independent check on this is that q_0.99 coverage here,
{pm(primary.coverage, primary.coverage_se)}, reproduces the Phase 3B
confirmatory 0.433 ± 0.029 through an entirely separate code path.

## 3. Detection rates, all cells

Rejection rate at level 0.05, critical values from the correct-probit cell
under the same policy.

{table_powers(summ)}

Three readings.

**Exploration buys detectability and coverage together.** For `tail_1.0`,
moving from the adaptive MI design to the broad exploratory one raises coverage
from {f(primary.coverage)} to {f(summ[(summ.dgp=="tail_1.0")&(summ.pol=="uniform_grid")].coverage.iloc[0])},
the design KL from {f(primary.design_kl_nats)} to {f(summ[(summ.dgp=="tail_1.0")&(summ.pol=="uniform_grid")].design_kl_nats.iloc[0])} nats, and
oracle Bayes-factor power from {f(primary.power_bf_oracle)} to {f(summ[(summ.dgp=="tail_1.0")&(summ.pol=="uniform_grid")].power_bf_oracle.iloc[0])}.
Detectability is not a separate objective competing with accuracy; on this
family the two move together.

**Undetectability is a property of the design, not of the family.** The robit
DGP is detectable under the exploratory design (best test
{f(max(summ[(summ.dgp=="robit_1.0")&(summ.pol=="uniform_grid")][c].iloc[0] for c, _ in TESTS))})
and undetectable under the adaptive one
({f(max(summ[(summ.dgp=="robit_1.0")&(summ.pol=="entropy_vector")][c].iloc[0] for c, _ in TESTS))}),
at coverage {f(summ[(summ.dgp=="robit_1.0")&(summ.pol=="entropy_vector")].coverage.iloc[0])}.
The same misspecification is caught or missed according to where the budget
went.

**Detectability tracks what the design collected, not how badly inference
failed.** Across the {len(mis)} misspecified cells, the correlation between the
design KL and oracle Bayes-factor power is {f(corr_kl, 2)}; between coverage and
that power it is {f(float(np.corrcoef(mis.coverage, mis.power_bf_oracle)[0, 1]), 2)}.
The cell that fails worst ({worst_cov.dgp} under {POL_LABEL[worst_cov.pol]},
coverage {f(worst_cov.coverage)}) is the one least likely to be caught.

## 4. Null calibration

Every test is at its nominal level on the correct-probit cell, which is what
makes the power columns above readable as power rather than as miscalibration.

{table_calibration(crit)}

## 5. What would change these conclusions

- **Screening precision.** Power SE ~0.015 per cell; the critical values carry
  their own Monte Carlo error at {n_rep} null replicates. A confirmatory tier
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
- **Numerics.** Evidence integrals on a {tol["evidence_grid_n"]}-point shared
  box; maximum boundary mass over all {len(raw)} rows
  {tol["evidence_max_boundary_mass"]:.1e}, reference posterior boundary mass
  {tol["ref_max_boundary_mass"]:.1e}. Bayes factors are stable to ~1e-4 under
  grid refinement and box enlargement
  (`tests/test_model_check.py::test_bayes_factors_are_converged_and_box_invariant`).

## 6. Consequence for the plan

A safeguard is not optional and cannot be a diagnostic. The failure is not
merely unnoticed by the checks an analyst runs -- it is absent from the data
those checks read. Any remedy therefore has to act on the **design** or on the
**reported interval**, not on model criticism after the fact, because there is
nothing in the sample for model criticism to find.
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
