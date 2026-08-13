"""Assemble every Phase 3 number and figure the atlas and report need."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.reporting import (  # noqa: E402
    plot_atlas,
    plot_design_distributions,
    plot_nooverlap_gap,
    plot_pseudo_truth,
)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
SUM = ROOT / "results" / "summaries"
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)


def section(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


def phase3a():
    p = RAW / "phase3a_retained.parquet"
    if not p.exists():
        print("3A missing")
        return
    d = pd.read_parquet(p)
    section("PHASE 3A - H1: paired particle vs reference on identical histories")
    rows = []
    for (arm, mech), g in d.groupby(["arm", "mechanism"]):
        for t in ("q0.5", "q0.95", "q0.99"):
            cd = g[f"{t}_cov_diff"].to_numpy()
            n = len(g)
            se = cd.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            rows.append({
                "arm": arm, "mechanism": mech, "target": t, "n": n,
                "cov_particle": g[f"{t}_p_covered"].mean(),
                "cov_reference": g[f"{t}_r_covered"].mean(),
                "paired_diff": cd.mean(), "se": se,
                "z": cd.mean() / se if se else np.nan,
                "n_discordant": int((cd != 0).sum()),
                "ess": g.particle_ess.mean(),
                "sd_ratio": g[f"{t}_sd_ratio"].mean(),
            })
    df = pd.DataFrame(rows)
    df.to_csv(SUM / "phase3a_h1.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    section("PHASE 3A - does no-overlap add beyond the (mu, sigma) stratum?")
    st = d[d.mechanism == "steep_in_support"]
    for t in ("q0.5", "q0.95", "q0.99"):
        a = st[st.arm == "no_overlap"][f"{t}_cov_diff"]
        b = st[st.arm == "overlap_control"][f"{t}_cov_diff"]
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"  {t}: no-overlap {a.mean():+.4f}  control {b.mean():+.4f}  "
              f"extra {diff:+.4f} +/- {se:.4f}  (z = {diff/se:.1f})")

    section("PHASE 3A - event rates (how common is this unconditionally?)")
    if (SUM / "phase3a_pilot_rates.csv").exists():
        pr = pd.read_csv(SUM / "phase3a_pilot_rates.csv")
        print(pr.pivot(index="sigma_true", columns="mu_true", values="rate")
              .to_string(float_format=lambda v: f"{v:5.2f}"))
        print(f"\npilot overall (uniform over that (mu,sigma) box): "
              f"{pr.n_no_overlap.sum() / pr.n.sum():.4f}")
    er = pd.read_csv(SUM / "phase3a_event_rates.csv")
    print("\nenrichment strata:")
    print(er.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    section("PHASE 3A - attribution: particle count and rejuvenation ablation")
    if (SUM / "phase3a_attribution.csv").exists():
        at = pd.read_csv(SUM / "phase3a_attribution.csv")
        sel = at[at.target == "q0.95"].sort_values(["sigma_true", "n_particles", "arm"])
        print(sel[["sigma_true", "arm", "n", "cov_particle", "cov_reference",
                   "cov_diff", "cov_diff_se", "abs_rank_diff", "sd_ratio", "ess",
                   "n_resamples"]].to_string(index=False,
                                             float_format=lambda v: f"{v:9.4f}"))
    print("\nfigure:", plot_nooverlap_gap(d))


def phase3b():
    p = SUM / "phase3b_screen_summary.csv"
    if not p.exists():
        print("3B screening missing")
        return None
    s = pd.read_csv(p)
    th = json.loads((ROOT / "configs" / "phase3_thresholds.json").read_text())
    section("PHASE 3B - frozen false-certainty thresholds")
    print(json.dumps(th["thresholds"], indent=2))
    print("hash:", th["hash"], "| from", th["source_cell"],
          "| n =", th["n_replicates"])

    ref = s[(s.backend == "ref") & (s.n == 50)]
    section("PHASE 3B BLOCK I - coverage of physical quantiles (reference, n=50)")
    b1 = ref[ref.block == "I"]
    for t in ("q0.95", "q0.99"):
        print(f"\n--- {t} coverage ---")
        print(b1[b1.target == t].pivot_table(index="dgp", columns="pol",
                                             values="coverage")
              .to_string(float_format=lambda v: f"{v:6.3f}"))
        print(f"--- {t} false certainty (misses AND width <= w*) ---")
        print(b1[b1.target == t].pivot_table(index="dgp", columns="pol",
                                             values="false_certainty")
              .to_string(float_format=lambda v: f"{v:6.3f}"))
        print(f"--- {t} mean interval width  (w* = "
              f"{th['thresholds'].get(f'{t}|n50', float('nan')):.3f}) ---")
        print(b1[b1.target == t].pivot_table(index="dgp", columns="pol",
                                             values="mean_width")
              .to_string(float_format=lambda v: f"{v:6.3f}"))

    section("PHASE 3B - H3 screen: any cell with coverage < 0.80 AND width <= w*?")
    cand = ref[(ref.coverage < 0.80) & (ref.mean_width <= ref.w_star)
               & ref.target.isin(["q0.95", "q0.99"])]
    if len(cand):
        print(cand[["block", "dgp", "pol", "prior_name", "scale", "target",
                    "coverage", "coverage_se", "mean_width", "w_star",
                    "false_certainty", "n_rep"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    else:
        print("  NONE at screening precision.")
        print("\n  closest cells by coverage (reference, q0.95/q0.99, all blocks):")
        print(ref[ref.target.isin(["q0.95", "q0.99"])]
              .nsmallest(12, "coverage")[["block", "dgp", "pol", "target", "coverage",
                                          "coverage_se", "mean_width", "w_star",
                                          "false_certainty", "rmse"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    section("PHASE 3B - H2 screen: adaptive vs non-adaptive false certainty (Block I)")
    rows = []
    for (dgp, t), g in b1.groupby(["dgp", "target"]):
        gg = g.set_index("pol")
        if "entropy_vector" in gg.index and "fixed_design" in gg.index:
            rows.append({
                "dgp": dgp, "target": t,
                "fc_adaptive": gg.loc["entropy_vector", "false_certainty"],
                "fc_fixed": gg.loc["fixed_design", "false_certainty"],
                "diff": gg.loc["entropy_vector", "false_certainty"]
                        - gg.loc["fixed_design", "false_certainty"],
                "cov_adaptive": gg.loc["entropy_vector", "coverage"],
                "cov_fixed": gg.loc["fixed_design", "coverage"],
                "se_approx": float(np.sqrt(gg.loc["entropy_vector", "false_certainty_se"]**2
                                           + gg.loc["fixed_design", "false_certainty_se"]**2)),
            })
    h2 = pd.DataFrame(rows).sort_values("diff", ascending=False)
    h2.to_csv(SUM / "phase3b_h2_screen.csv", index=False)
    print(h2[h2.target != "q0.5"].to_string(index=False,
                                            float_format=lambda v: f"{v:8.3f}"))

    section("PHASE 3B BLOCK II - design-support failure (reference, n=50)")
    b2 = ref[ref.block == "II"]
    if len(b2):
        print(b2[b2.target.isin(["q0.95", "q0.99"])][
            ["dgp", "pol", "target", "target_true", "coverage", "coverage_se",
             "bias", "rmse", "mean_width", "false_certainty", "dangerous_error",
             "target_outside_sampled", "unresolved_rate"]]
            .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    section("PHASE 3B - sub-blocks: curve scale (Ib) and prior (Ic)")
    for blk, lab in (("Ib", "curve scale"), ("Ic", "prior")):
        sub = ref[ref.block == blk]
        if len(sub):
            print(f"\n--- {blk}: {lab} ---")
            print(sub[sub.target == "q0.95"][
                ["dgp", "pol", "prior_name", "scale", "coverage", "coverage_se",
                 "mean_width", "false_certainty", "rmse"]]
                .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    section("PHASE 3B - particle vs reference overlay (is any of this numerical?)")
    both = s[(s.n == 50)].pivot_table(index=["block", "dgp", "pol", "target"],
                                      columns="backend", values="coverage")
    both = both.dropna()
    both["diff"] = both["particle"] - both["ref"]
    print(f"particle - reference coverage: mean {both['diff'].mean():+.4f}, "
          f"p05 {both['diff'].quantile(0.05):+.4f}, p95 {both['diff'].quantile(0.95):+.4f}, "
          f"max |diff| {both['diff'].abs().max():.4f} over {len(both)} cells")
    print("\nlargest disagreements:")
    print(both.reindex(both["diff"].abs().sort_values(ascending=False).index)
          .head(8).to_string(float_format=lambda v: f"{v:8.4f}"))

    section("PHASE 3B - degeneracies (nothing dropped)")
    man = json.loads((ROOT / "results" / "manifests" / "phase3b_screen.json").read_text())
    print(json.dumps(man["degeneracy_counts"], indent=2))
    print("failures:", man["n_failures"])

    figs = [plot_atlas(s, block="I", backend="ref", n=50)]
    if (ref.block == "II").any():
        figs.append(plot_atlas(s, block="II", backend="ref", n=50))
    print("\nfigures:", [str(f) for f in figs if f])
    return s


def pseudo_truth():
    p = SUM / "phase3b_pseudo_truth.csv"
    if not p.exists():
        print("pseudo-truth missing")
        return
    d = pd.read_csv(p)
    sp = pd.read_csv(SUM / "phase3b_pseudo_truth_spread.csv")
    section("PHASE 3B - H4: policy-dependent pseudo-true quantiles")
    sel = sp[(sp.block == "I") & (sp.scale == "s3")].sort_values("spread",
                                                                 ascending=False)
    print(sel[["dgp", "target", "physical", "pseudo_min", "pseudo_max", "spread",
               "spread_se", "policy_at_min", "policy_at_max",
               "max_abs_pseudo_minus_physical"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    q95 = sel[sel.target == "q0.95"]
    n_big = int((q95.spread >= 1.5).sum())
    print(f"\nH4 threshold is a spread >= 1.5 stress units (0.5 sigma_0): "
          f"{n_big}/{len(q95)} Block I DGPs meet it for q0.95")

    section("PHASE 3B - pseudo-true vs physical (bias the posterior converges to)")
    print(d[(d.block == "I") & (d.scale == "s3")][
        ["dgp", "pol", "pseudo_mu", "pseudo_sigma", "pseudo_q0.95",
         "physical_q0.95", "pseudo_minus_physical_q0.95", "pseudo_q0.95_se",
         "design_q05", "design_q50", "design_q95"]]
        .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    figs = [plot_pseudo_truth(d, block="I", target="q0.95"),
            plot_pseudo_truth(d, block="I", target="q0.99",
                              name="phase3b_pseudo_truth_I_q0.99")]
    dpath = RAW / "phase3b_designs.parquet"
    if dpath.exists():
        designs = pd.read_parquet(dpath)
        cells = [c for c in ("I|tail_1.0|entropy_vector|cal|s3",
                             "I|tail_1.0|fixed_design|cal|s3",
                             "I|tail_1.0|uniform_grid|cal|s3",
                             "I|probit|entropy_vector|cal|s3")
                 if c in set(designs.cell)]
        if cells:
            figs.append(plot_design_distributions(designs, cells))
    print("\nfigures:", [str(f) for f in figs if f])


def reference_audit():
    p = SUM / "phase3_reference_audit_summary.csv"
    if p.exists():
        section("PHASE 3 - reference adequacy audit")
        print(pd.read_csv(p).to_string(index=False,
                                       float_format=lambda v: f"{v:10.5f}"))


def main():
    reference_audit()
    phase3a()
    phase3b()
    pseudo_truth()
    section("done")


if __name__ == "__main__":
    main()
