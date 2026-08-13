# thesis-reliability

Implementation of **Phases 0–3** of *When Adaptive Posteriors Become Confidently
Wrong: Calibrated Quantile Inference in Sequential Sensitivity Experiments*.

Phases 4–7 (theory, the online reliability map, safeguards and extensions) are
**not** implemented here. Each phase gate requires a validation report first:
`manuscript/phase2_validation_report.md` and
`manuscript/phase3_validation_report.md`.

Phase 3 is pre-registered. `manuscript/phase3_protocol.md` was frozen at commit
`2c452d7`, before any Phase 3 experiment ran; its version history records every
subsequent deviation and the reason, and no confirmatory definition was changed
after results were seen.

---

Working conventions — invariants, the provenance contract, statistical rules,
and the gate ritual — are in [`CLAUDE.md`](CLAUDE.md). Claim status lives in
[`docs/claims.md`](docs/claims.md), choices in
[`docs/decisions.md`](docs/decisions.md), deviations in
[`docs/discrepancies.md`](docs/discrepancies.md).

## What is here

| Phase | Deliverable | Where |
|---|---|---|
| 0 | Likelihood-factorisation proposition, two-quantile result, revised thesis question | `manuscript/phase0_note.md`, `src/factorization.py` |
| 1 | Reproduction of Rotem's baseline (Tables 1, 2, 3 and the Bruceton columns of 6–7), stimulus trajectories, the utility surface $U_p(r;H_n)$ | `experiments/phase1_*.py` |
| 2 | SBC with the adaptive policy in the loop, fixed-$\theta$ coverage maps, path-stratified coverage | `experiments/phase2_*.py` |
| 3 | Pre-registered protocol; powered no-overlap investigation; misspecification failure atlas; policy-dependent pseudo-truth | `manuscript/phase3_*.md`, `experiments/phase3*.py` |

## Layout

```
src/
  response_models.py   probit fitted family; true curves (probit, Beta-CDF)
  priors.py            Rotem's four prior settings, sampler + density
  posterior_grid.py    refined-grid REFERENCE posterior on (mu, log sigma)
  rotem_particles.py   Rotem's weighted-particle posterior + rejuvenation
  policies.py          entropy/MI designs, Dror-Steinberg, Bruceton, controls
  utilities.py         mutual information (vector and scalar), U_p(r) surface
  factorization.py     the Phase 0 invariant, made checkable
  simulator.py         sequential experiment driver, CRN, slicing
  calibration.py       SBC, coverage, stratification
  diagnostics.py       observable path summaries (inputs to the Phase 5 map)
  estimators.py        probit MLE, Dixon-Mood
  decision_loss.py     asymmetric threshold loss, false-certainty rate
  manifest.py          run provenance and Monte Carlo standard errors
  curve_families.py    matched monotone alternatives + targeted tail perturbation
  pseudo_truth.py      probit KL projection under a design distribution
  reporting.py         figures and the comparison against published tables
docs/
  discrepancies.md         every departure from the thesis, with reasons
  rotem_published.csv      the published MSE tables, transcribed
```

## Running

```bash
pip install -e ".[dev]"
python -m pytest                       # required tests 1-10, numerics, provenance
python -m pytest -m slow               # the slower high-accuracy checks
python tools/manifest_lint.py --strict # the provenance contract
```

Experiments run from a config, so a result is never reproducible only from a
shell history:

```bash
python experiments/run.py configs/experiments/phase1_baseline.json
python experiments/run.py configs/experiments/phase2_sbc.json
python experiments/run.py configs/experiments/phase3b_confirm.json
python experiments/build_phase3_report.py
```

`--replicates`, `--seed-base`, `--crn-seed`, `--workers` and `--note` override
the config for a pilot; `--dry-run` resolves the invocation without running it.
Scripts also still take their flags directly
(`python experiments/phase1_baseline.py --replicates 250`), and a bare
positional count is accepted for backward compatibility.

**Runs start from a clean tree.** A run from an uncommitted tree is refused,
because a result produced from one cannot be tied to a state of the source —
which has already cost this project the Phase 3A/3B interval-width comparison
(`docs/discrepancies.md` D19). `--allow-dirty` waives the refusal for pilots
and records `provenance.allow_dirty` in the manifest.

`N_WORKERS` and `CHUNK` control parallelism. Every script writes raw
per-replicate rows to `results/raw/`, summaries to `results/summaries/`, and a
manifest to `results/manifests/` recording the run id, config path and hash,
git commit and dirty state, seed base and seeding rule, completed replicates,
Monte Carlo standard errors, tolerances, failure counts, degeneracy counts and
the role-tagged artefacts produced.

> **Open gap (D21).** 28 of the 31 artefacts referenced by manifests are absent
> from this repository, including every raw parquet and the summaries that are
> the named evidence for C3, C5, C6 and C7. Until that is resolved, the results
> here cannot be independently checked. See `docs/claims.md`.

## Rules the code enforces

**Nothing is discarded.** All-zero and all-one paths, runs with no overlapping
pattern, collapsed particle ESS, non-finite MLEs and raised exceptions are
recorded with flags and counted in the manifest. They are outcomes of the
method, not noise to be cleaned away.

**Every run is pinned and identified.** One `run_id` per run, used as the
manifest filename stem so manifest, raw table and summaries join on one key;
`seed_base` and `crn_seed` are authoritative and the seeding rule is recorded
verbatim. There is no `seed_range` — the field existed in earlier manifests and
was never used by the code. `tools/manifest_lint.py` enforces this.

**Every optimisation is exactly equivalent.** The hot path uses a
generation-keyed cache, an (L, N) memory layout, and an algebraic
simplification of the scalar mutual-information quadrature. Each is checked
against the unoptimised formulation in `tests/test_numerics.py`, to $10^{-12}$
or exactly.

## The required tests

| # | Requirement | Test |
|---|---|---|
| 1 | Likelihood proportionality, deterministic and randomised policies | `test_factorization.py` (plus a non-ignorable negative control) |
| 2 | Sequential updates vs recomputation from the full history | `test_factorization.py::test_sequential_updates_match_full_recomputation` |
| 3 | $(\mu,\sigma)\leftrightarrow(q_{p_1},q_{p_2})$ round trip | `test_reparameterization.py::test_quantile_roundtrip` |
| 4 | MI invariance under the two parameterisations | `test_reparameterization.py::test_vector_mi_invariant_under_two_quantile_reparameterization` |
| 5 | Grid normalisation and refinement convergence | `test_posterior_grid.py` |
| 6 | Posterior recovery in a brute-force case | `test_posterior_grid.py::test_posterior_matches_brute_force_importance_sampling` |
| 7 | Particle weights, ESS, resampling, rejuvenation | `test_particles.py` |
| 8 | SBC uniformity, non-adaptive correct model | `test_calibration.py::test_reference_posterior_passes_sbc[uniform_grid-...]` |
| 9 | SBC uniformity, adaptive correct model | `test_calibration.py::test_reference_posterior_passes_sbc[entropy_vector-...]` |
| 10 | Deterministic replay of a saved configuration | `test_reproducibility.py` |
