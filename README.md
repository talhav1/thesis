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
python -m pytest                      # required tests 1-10 plus numerics
python -m pytest -m slow               # the slower high-accuracy checks

python experiments/phase1_baseline.py  250   # Rotem reproduction
python experiments/phase1_horizon.py    120   # monotone-in-n check
python experiments/phase1_utility_surface.py
python experiments/phase2_sbc.py        1000  # SBC, both backends
python experiments/phase2_coverage_map.py 200 # fixed-theta coverage

python experiments/phase3a_pilot.py       30    # where no-overlap paths come from
python experiments/phase3a_nooverlap.py   1.0   # powered H1, stratified enrichment
python experiments/phase3a_attribution.py 50    # particle-count / rejuvenation ablation
python experiments/phase3_reference_audit.py 24 # reference adequacy
python experiments/phase3b_screen.py      30    # failure atlas, screening tier
python experiments/phase3b_pseudotruth.py 100   # policy-dependent pseudo-truth
python experiments/phase3b_confirm.py     300   # confirmatory tier
python experiments/build_phase3_report.py
```

`N_WORKERS` and `CHUNK` control parallelism. Every script writes raw
per-replicate rows to `results/raw/`, summaries to `results/summaries/`, and a
manifest to `results/manifests/` recording the config hash, git commit, seed
range, completed replicates, Monte Carlo standard errors, tolerances, failure
counts, degeneracy counts and the artefacts produced.

## Two rules the code enforces

**Nothing is discarded.** All-zero and all-one paths, runs with no overlapping
pattern, collapsed particle ESS, non-finite MLEs and raised exceptions are
recorded with flags and counted in the manifest. They are outcomes of the
method, not noise to be cleaned away.

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
