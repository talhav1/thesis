"""Assemble every number and figure the Phase 2 validation report needs.

Run after the Phase 1 and Phase 2 experiment scripts.  Reads only from
`results/`, writes summaries and figures back into `results/`, and prints a
compact digest to stdout so the report can be written from one place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.reporting import (  # noqa: E402
    compare_to_published,
    plot_coverage_map,
    plot_rank_agreement,
    plot_reproduction,
    plot_sbc,
    plot_trajectories,
    plot_utility_surface,
)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
SUM = ROOT / "results" / "summaries"
pd.set_option("display.width", 200)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def phase1():
    rawp = RAW / "phase1_baseline_raw.parquet"
    if not rawp.exists():
        print("phase1 raw missing; skipping")
        return None
    # Recompute the summary from the raw rows rather than trusting the CSV the
    # experiment script wrote: the raw file is the primary record, and this
    # keeps the report reproducible from it alone.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase1_baseline import summarise

    summary = summarise(pd.read_parquet(rawp))
    summary.to_csv(SUM / "phase1_baseline_mse.csv", index=False)
    cmp = compare_to_published(summary)
    cmp.to_csv(SUM / "phase1_vs_published.csv", index=False)

    section("PHASE 1 - reproduction of Rotem's published MSEs (n = 30, Bayes)")
    ok = cmp.dropna(subset=["mse_reproduced"])
    print(f"cells compared: {len(ok)}")
    print(f"within 2 combined SE: {int(ok.within_2se.sum())}/{len(ok)} "
          f"({ok.within_2se.mean():.1%})")
    print(f"median |ratio - 1|: {np.median((ok.ratio - 1).abs()):.3f}")
    print(f"median MC SE / MSE: {np.median(ok.mse_se / ok.mse_reproduced):.3f}")
    print("\nlargest discrepancies (|z_combined|):")
    print(ok.reindex(ok.z_combined.abs().sort_values(ascending=False).index)
          .head(12)[["setting", "policy", "target", "mse_published",
                     "mse_reproduced", "mse_se", "z_combined"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    fig1 = plot_reproduction(summary)
    fig2, ranks = plot_rank_agreement(summary)
    ranks.to_csv(SUM / "phase1_rank_agreement.csv", index=False)
    section("PHASE 1 - agreement in method ordering")
    print(ranks.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))
    print(f"\nmean rank correlation: {ranks.spearman.mean():.3f}; "
          f"same best method in {ranks.same_best.mean():.1%} of cells")

    # paired comparisons exploit the common random numbers
    raw = pd.read_parquet(RAW / "phase1_baseline_raw.parquet")
    raw["setting_only"] = raw.setting.str.split("|").str[0]
    rows = []
    for (setting, n), g in raw[raw.n == 30].groupby(["setting_only", "n"]):
        for t in ("mu", "sigma", "q0.95", "q0.99"):
            col = f"{t}_err"
            if col not in g:
                continue
            piv = g.pivot_table(index="replicate", columns="policy", values=col)
            piv = piv.dropna()
            if "entropy_vector" not in piv or "dror_steinberg" not in piv:
                continue
            d = piv["entropy_vector"] ** 2 - piv["dror_steinberg"] ** 2
            rows.append({
                "setting": setting, "target": t,
                "paired_mse_diff_entropy_minus_ds": float(d.mean()),
                "paired_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                "unpaired_se": float(np.sqrt(
                    (piv["entropy_vector"] ** 2).var(ddof=1) / len(piv)
                    + (piv["dror_steinberg"] ** 2).var(ddof=1) / len(piv))),
                "n_pairs": len(piv),
            })
    paired = pd.DataFrame(rows)
    paired["se_reduction"] = 1 - paired.paired_se / paired.unpaired_se
    paired.to_csv(SUM / "phase1_paired_entropy_vs_ds.csv", index=False)
    section("PHASE 1 - paired entropy(mu,sigma) vs DS (common random numbers)")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print(f"\nmedian SE reduction from pairing: {paired.se_reduction.median():.1%}")

    section("PHASE 1 - degeneracy counts (nothing dropped)")
    man = json.loads((ROOT / "results" / "manifests" / "phase1_baseline.json").read_text())
    print(json.dumps(man["degeneracy_counts"], indent=2))
    print(f"failures: {man['n_failures']}")

    tp = RAW / "phase1_trajectories.parquet"
    figs = [fig1, fig2]
    if tp.exists():
        traj = pd.read_parquet(tp)
        keys = [k for k in traj.setting.unique()
                if k.startswith("probit_ind_well") or k.startswith("beta_ind_well")]
        keys = [k for k in keys if k.endswith(("entropy_vector", "entropy_q0.95",
                                               "bruceton"))][:3]
        if keys:
            figs.append(plot_trajectories(traj, keys))
    print("\nfigures:", [str(f) for f in figs])
    return summary


def phase1_utility():
    p = SUM / "phase1_utility_surface.csv"
    if not p.exists():
        print("utility surface missing; skipping")
        return
    df = pd.read_csv(p)
    section("PHASE 1 - quantile-indexed utility surface U_p(r; H_n)")
    peaks = pd.read_csv(SUM / "phase1_utility_surface_peaks.csv")
    print(peaks.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print(f"\nquadrature density integral in "
          f"[{df.quad_density_integral.min():.4f}, "
          f"{df.quad_density_integral.max():.4f}]")
    print("figure:", plot_utility_surface(df))


def phase2():
    p = SUM / "phase2_sbc_uniformity.csv"
    if not p.exists():
        print("phase2 SBC missing; skipping")
        return
    rep = pd.read_csv(p)
    section("PHASE 2 - SBC uniformity (adaptive policy inside the simulator)")
    print(rep.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    ref = rep[rep.backend == "ref"]
    par = rep[rep.backend == "particle"]
    print(f"\nreference: min KS p = {ref.ks_p.min():.4f}; "
          f"{int((ref.ks_p > 0.05).sum())}/{len(ref)} pass at alpha=0.05")
    print(f"particle : min KS p = {par.ks_p.min():.4f}; "
          f"{int((par.ks_p > 0.05).sum())}/{len(par)} pass at alpha=0.05")
    print(f"reference max KS D = {ref.ks_stat.max():.4f} "
          f"(95% band {ref.ks_band_95.iloc[0]:.4f})")
    print(f"particle  max KS D = {par.ks_stat.max():.4f}")

    raw = pd.read_parquet(RAW / "phase2_sbc_raw.parquet")
    print("figure:", plot_sbc(raw, ["mu", "sigma", "q0.95", "q0.99"]))

    sp = SUM / "phase2_coverage_stratified.csv"
    if sp.exists():
        strat = pd.read_csv(sp)
        section("PHASE 2 - coverage stratified by observable path status (SBC draws)")
        sel = strat[(strat.stratum == "overlap_status")
                    & strat.target.isin(["q0.95", "q0.99"])]
        print(sel.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))


def phase2_coverage():
    p = SUM / "phase2_coverage_map.csv"
    if not p.exists():
        print("phase2 coverage map missing; skipping")
        return
    cov = pd.read_csv(p)
    section("PHASE 2 - fixed-theta coverage map")
    sel = cov[cov.target.isin(["q0.5", "q0.95", "q0.99"])]
    print(sel.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    figs = [plot_coverage_map(cov, backend="particle")]
    if (cov.backend == "ref").any():
        figs.append(plot_coverage_map(cov, backend="ref",
                                      name="phase2_coverage_map_ref"))
    print("figures:", [str(f) for f in figs])

    fp = SUM / "phase2_false_certainty.csv"
    if fp.exists():
        fc = pd.read_csv(fp)
        section("PHASE 2 - false-certainty rate (narrow AND wrong)")
        print(fc[fc.target.isin(["q0.95", "q0.99"])]
              .to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    section("PHASE 2 - particle vs reference posterior on identical histories")
    raw = pd.read_parquet(RAW / "phase2_coverage_map_raw.parquet")
    rows = []
    for t in ("mu", "sigma", "q0.95", "q0.99"):
        d, r = f"{t}_particle_vs_ref_mean", f"{t}_particle_vs_ref_sd_ratio"
        if d not in raw:
            continue
        rows.append({
            "target": t,
            "mean_shift_mean": float(raw[d].mean()),
            "mean_shift_sd": float(raw[d].std()),
            "mean_shift_p95_abs": float(raw[d].abs().quantile(0.95)),
            "ref_sd": float(raw[f"{t}_ref_sd"].mean()),
            "shift_in_ref_sd_units": float(raw[d].abs().mean()
                                           / raw[f"{t}_ref_sd"].mean()),
            "sd_ratio_mean": float(raw[r].mean()),
            "sd_ratio_p05": float(raw[r].quantile(0.05)),
            "sd_ratio_p95": float(raw[r].quantile(0.95)),
        })
    fid = pd.DataFrame(rows)
    fid.to_csv(SUM / "phase2_computational_fidelity.csv", index=False)
    print(fid.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))


def main():
    phase1()
    phase1_utility()
    phase2()
    phase2_coverage()
    section("done")


if __name__ == "__main__":
    main()
