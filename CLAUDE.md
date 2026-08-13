# CLAUDE.md — working contract for this repository

Read this before changing code, running an experiment, or writing a report.
It is the standing instruction set; `Thesis_Implementation_Plan.md` is the
research plan, and where the two disagree, the plan wins on *what* to do and
this file wins on *how*.

---

## 1. What this project is

A statistical thesis on **quantile reliability in Bayesian adaptive sensitivity
experiments**. Working title: *When Adaptive Posteriors Become Confidently
Wrong*. The question is when posterior credible intervals and threshold
decisions about response quantiles — especially upper-tail quantiles
$q_{0.95}, q_{0.99}$ — are trustworthy, and how the design can be made safer
under model misspecification or weak exploration.

The baseline being reproduced and stress-tested is Rotem's thesis: a probit
location-scale model, $p_\theta(x) = \Phi((x-\mu)/\sigma)$, with mutual-
information stimulus selection and a weighted-prior-particle posterior.

**Current state.** Phases 0–3 complete. Phase 3 established tail false
certainty under adaptive design as the strongest result in the project. Phases
4–7 (theory, online reliability map, safeguards, extensions) not started.
`docs/claims.md` is the authoritative status of every claim.

---

## 2. Hard invariants

These are properties of the mathematics, not of the implementation. If a test
covering one fails, the implementation is wrong. **Never** fix such a failure
by relaxing a tolerance, marking a test `xfail`, or narrowing its scope.

1. **Likelihood factorisation.** Under a predictable, parameter-free
   (ignorable) design policy, $\log p_\theta(H_n) - \sum_i \log p_\theta(y_i \mid x_i)$
   is constant in $\theta$. Enforced to $<10^{-9}$ over a 72-point grid, with
   total variation between the two posteriors $<10^{-12}$.
2. **The negative control must keep failing.** A non-ignorable oracle policy
   must *break* invariant 1 (residual spread $>1$ nat, TV $>0.01$). Without it,
   an implementation that simply drops the design term passes vacuously. Any
   change that makes the negative control pass is a regression.
3. **Two-quantile reparameterisation.** $(q_{p_1}, q_{p_2}) \leftrightarrow (\mu,\sigma)$
   round-trips to $<10^{-10}$ for $p_1 \neq p_2$, and mutual information is
   invariant under it. One quantile is *strictly* less informative.
4. **Reference posterior calibration.** Under a correct probit model the
   reference grid posterior passes SBC for every target under both an adaptive
   and a non-adaptive design.

`tests/test_factorization.py`, `test_reparameterization.py`,
`test_calibration.py`, `test_posterior_grid.py`.

---

## 3. Provenance contract

Every experimental result records: config file and its hash; source commit;
seed base and the seeding rule; completed replicates; Monte Carlo standard
errors; inference backend and numerical tolerances; failure and exception
counts; and the exact tables and figures it produced.

Concretely, in this repo:

- **Runs start from a clean tree.** `src/provenance.require_clean_tree` refuses
  otherwise. `--allow-dirty` waives it and records
  `provenance.allow_dirty: true` in the manifest. Use the waiver for pilots,
  never for a run that will be cited.
- **One `run_id` per run**, minted by `src/provenance.new_run_id` and used as
  the manifest filename stem, so manifest ↔ raw ↔ summaries join on one key.
- **`seed_base` and `crn_seed` are authoritative.** There is no contiguous seed
  range; the code seeds with `default_rng([seed_base, replicate])`. Never
  reintroduce a `seed_range` field. `src/provenance.SEED_RULE` states the
  scheme and a test asserts it matches the simulator.
- **`n_replicates_completed` means replicates.** If a script counts output
  rows, it must set `replicates_unit="rows"`. Two legacy manifests silently
  counted rows; that is what the field exists to prevent.
- **Artifacts are role-tagged and repo-relative.** `{"role": ..., "path": ...}`.
- **Nothing in `results/` without a manifest.** `experiments/_runner.py` is the
  only sanctioned writer.
- **`python tools/manifest_lint.py --strict` must pass** before a gate.

**Why this is not bureaucratic.** Every manifest written before this contract
records a `-dirty` commit. The bisection settings in
`GridPosterior._invert_cdf` changed inside the same commit that contains the
Phase 3A outputs, so the manifest cannot establish which settings Phase 3A
used, and Phase 3A/3B interval widths are permanently non-comparable. That is
one real result lost to a missing three-line guard.

---

## 4. Statistical rules

- **Never silently drop** failed runs, all-zero or all-one paths, or numerical
  degeneracies. They are primary outcomes. Count them and put them in
  `degeneracy_counts`.
- **Every reported number carries a Monte Carlo standard error.** "Reproduced"
  means agreement within combined MC uncertainty, stated, not eyeballed.
- **State precision tier.** *Confirmatory* (pre-registered replicate count) vs
  *screening* (30-ish replicates, coverage SE 0.03–0.11). A screening number
  must never be written as if it were confirmatory.
- **Physical quantities are the estimands.** Under misspecification, `mu` means
  the true curve's median and `sigma` its standard deviation — not the fitted
  probit's parameters.
- **Keep the two error sources separate.** Computational posterior error
  (particle/rejuvenation) and response-model misspecification are different
  chapters. Do not let a result attribute one to the other.
- **Coverage is read off the rank statistic**, $\alpha/2 < P(Q < Q_{true} \mid \text{data}) < 1-\alpha/2$,
  not by forming an interval and testing containment, so the coverage layer and
  the SBC layer stay consistent by construction.

---

## 5. What not to do

From the plan, still binding:

- Do not describe Rotem's likelihood as an independence approximation. It is
  exact under an ignorable policy. This question is **closed**.
- Do not make simulation-based inference the methodological framework; the
  likelihood is available here.
- Do not interpret every fixed-parameter coverage deviation as evidence that
  Bayesian inference is invalid.
- Do not mix computational posterior error with response-model
  misspecification.
- Do not start a general robust-Bayes method before the failure mechanism is
  established in the two-parameter problem.
- Do not promote the AI-safety/LLM application to the thesis core.

Added from experience:

- Do not rewrite history in `docs/discrepancies.md` or `docs/decisions.md`.
  They are append-only. A superseded entry is marked superseded, not deleted.
- Do not correct a report to agree with a results file, or a results file to
  agree with a report. The results file wins and the disagreement gets an
  entry.
- Do not report a number without checking which tier produced it.

---

## 6. Layout and ownership

```
src/           library code. No I/O to results/.
experiments/   run scripts. run.py is the config-driven entry point;
               _cli.py the shared flags; _runner.py the only results/ writer.
configs/       experiments/*.json = one config per published run.
results/       raw/ (parquet), summaries/ (csv), figures/, manifests/.
               Written by runners only. Never hand-edited.
docs/          discrepancies.md, claims.md, decisions.md, gates.md — all
               append-only ledgers.
manuscript/    advisor-facing write-ups. Generated from results/, never typed.
tools/         manifest_lint.py and other repo hygiene.
tests/         104+ tests. `pytest` excludes slow by default.
```

### The three ledgers, and what belongs in which

| Ledger | Contains | Example |
|---|---|---|
| `docs/discrepancies.md` | deviations from Rotem or from the plan, D-numbered | D13 `entropy_sigma` does not reproduce |
| `docs/claims.md` | every claim with status, evidence file, precision tier | C2 Established, C6 Preliminary |
| `docs/decisions.md` | statistical choices, with rationale and date | $w^\ast$ frozen from one cell |

---

## 7. Running things

```bash
pytest                                              # fast suite
pytest -m slow                                      # the long checks
python experiments/run.py configs/experiments/phase1_baseline.json
python experiments/run.py configs/.../x.json --replicates 5 --allow-dirty  # pilot
python tools/manifest_lint.py --strict
```

Long runs (Phase 1 baseline ≈ 53 min, Phase 3B screen ≈ 67 min) should be
launched detached, not held open in an interactive session.

---

## 8. Gate ritual

At every phase boundary, in this order — the full version is `docs/gates.md`:

1. Freeze the code and **commit**; the tree must be clean.
2. **Tag** it `phaseN-gate`.
3. Run from configs. Manifests record the tag.
4. Write the validation report **from the summaries**, not from memory.
5. **Audit in a fresh session**: check every claim against `results/`; the
   results file wins.
6. Append to `docs/discrepancies.md` and update `docs/claims.md`.
7. Only then take it to the advisor.

Step 5 is the one that catches overclaiming, and it works because the auditing
session has no draft to defend.
