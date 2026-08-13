"""Figures and comparison tables for the Phase 0-2 validation report.

Deliberately plain matplotlib: these are diagnostics for the advisor meeting,
not publication graphics.  Every figure is written from a results file that has
a manifest, and the figure's source file is recorded in that manifest.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"
PUBLISHED = ROOT / "docs" / "rotem_published.csv"

POLICY_ORDER = ["dror_steinberg", "entropy_vector", "entropy_mu", "entropy_sigma",
                "entropy_q0.95", "bruceton"]
POLICY_LABEL = {
    "dror_steinberg": "DS",
    "entropy_vector": r"Ent$(\mu,\sigma)$",
    "entropy_mu": r"Ent$(\mu)$",
    "entropy_sigma": r"Ent$(\sigma)$",
    "entropy_q0.95": r"Ent$(x_{0.95})$",
    "bruceton": "Bruceton",
}


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Phase 1: reproduction of Rotem's tables
# --------------------------------------------------------------------------


def compare_to_published(summary: pd.DataFrame) -> pd.DataFrame:
    """Join reproduced MSEs to the published values and score the agreement.

    `z` is the discrepancy in units of *our* Monte Carlo standard error.  It is
    a lower bound on the discrepancy in units of the combined error, because
    the thesis does not report the Monte Carlo error of its own S = 500 runs;
    the `z_combined` column assumes the published cell carries a comparable SE,
    which is the fairest assumption available.
    """
    pub = pd.read_csv(PUBLISHED)
    ours = summary[(summary.n == 30) & (summary.estimator == "bayes")][
        ["setting", "policy", "target", "mse", "mse_se", "n_total"]
    ]
    m = pub.merge(ours, on=["setting", "policy", "target"],
                  suffixes=("_pub", "_ours"), how="left")
    m = m.rename(columns={"mse_pub": "mse_published", "mse_ours": "mse_reproduced"})
    m["ratio"] = m.mse_reproduced / m.mse_published
    m["diff"] = m.mse_reproduced - m.mse_published
    m["z"] = m["diff"] / m.mse_se
    m["z_combined"] = m["diff"] / (m.mse_se * np.sqrt(2))
    m["within_2se"] = m.z_combined.abs() < 2
    return m


def plot_reproduction(summary: pd.DataFrame, name="phase1_reproduction"):
    m = compare_to_published(summary)
    settings = sorted(m.setting.unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    for ax, setting in zip(axes.ravel(), settings):
        sel = m[m.setting == setting]
        targets = sorted(sel.target.unique())
        width = 0.8 / max(len(targets), 1)
        for i, t in enumerate(targets):
            s = sel[sel.target == t].set_index("policy").reindex(POLICY_ORDER).dropna(
                subset=["mse_published"])
            xs = np.arange(len(s)) + i * width
            ax.bar(xs, s.mse_published, width * 0.45, color="0.65",
                   label="Rotem" if i == 0 else None)
            ax.bar(xs + width * 0.45, s.mse_reproduced, width * 0.45,
                   yerr=1.96 * s.mse_se, color="C0", ecolor="0.2", capsize=1.2,
                   label="reproduced" if i == 0 else None)
        s0 = sel[sel.target == targets[0]].set_index("policy").reindex(
            POLICY_ORDER).dropna(subset=["mse_published"])
        ax.set_xticks(np.arange(len(s0)) + 0.4)
        ax.set_xticklabels([POLICY_LABEL.get(p, p) for p in s0.index],
                           rotation=30, fontsize=8)
        ax.set_title(setting, fontsize=10)
        ax.set_ylabel("MSE")
        if setting == settings[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Phase 1: Rotem's published MSE vs reproduction (n = 30, "
                 "error bars = 95% Monte Carlo interval)", fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


def plot_rank_agreement(summary: pd.DataFrame, name="phase1_rank_agreement"):
    """Does the reproduction preserve the *ordering* of methods per target?"""
    m = compare_to_published(summary).dropna(subset=["mse_reproduced"])
    rows = []
    for (setting, target), g in m.groupby(["setting", "target"]):
        if len(g) < 3:
            continue
        rp = g.mse_published.rank()
        ro = g.mse_reproduced.rank()
        rows.append({
            "setting": setting, "target": target,
            "spearman": float(np.corrcoef(rp, ro)[0, 1]),
            "same_best": g.loc[g.mse_published.idxmin(), "policy"]
                         == g.loc[g.mse_reproduced.idxmin(), "policy"],
            "n_methods": len(g),
        })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    piv = df.pivot(index="target", columns="setting", values="spearman")
    im = ax.imshow(piv.values, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="rank correlation with Rotem")
    ax.set_title("Agreement in method ordering, per setting and target")
    fig.tight_layout()
    return _save(fig, name), df


# --------------------------------------------------------------------------
# Phase 1: trajectories and the utility surface
# --------------------------------------------------------------------------


def plot_trajectories(traj: pd.DataFrame, settings, name="phase1_trajectories",
                      n_paths=6):
    fig, axes = plt.subplots(1, len(settings), figsize=(4.4 * len(settings), 4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, settings):
        sel = traj[traj.setting == key].head(n_paths)
        for _, row in sel.iterrows():
            x = np.asarray(row["x"], dtype=float)
            y = np.asarray(row["y"], dtype=int)
            steps = np.arange(1, len(x) + 1)
            ax.plot(steps, x, color="0.75", lw=0.8, zorder=1)
            ax.scatter(steps[y == 1], x[y == 1], s=16, marker="^", color="C3",
                       zorder=2)
            ax.scatter(steps[y == 0], x[y == 0], s=16, marker="v", color="C0",
                       zorder=2)
        ax.axhline(30.0, color="k", ls="--", lw=0.8)
        ax.axhline(34.934, color="C2", ls=":", lw=1.0)
        ax.set_title(key.replace("|", "\n"), fontsize=9)
        ax.set_xlabel("step")
    axes[0].set_ylabel("stimulus")
    fig.suptitle("Selected stimulus trajectories "
                 "(up = response, down = non-response; dashed $\\mu$, dotted $x_{0.95}$)",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, name)


def plot_utility_surface(df: pd.DataFrame, name="phase1_utility_surface"):
    ns = sorted(df.n.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(4.6 * len(ns), 4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, ns):
        g = df[df.n == n]
        for p, gg in g.groupby("target_p"):
            gg = gg.sort_values("stimulus_r")
            ax.plot(gg.stimulus_r, gg.utility, marker="o", ms=3,
                    label=f"$p={p}$")
            best = gg.loc[gg.utility.idxmax()]
            ax.scatter([best.stimulus_r], [best.utility], s=60,
                       facecolors="none", edgecolors="k", zorder=5)
        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_xlabel("stimulated response level $r$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("$U_p(r; H_n)$   [nats]")
    axes[0].legend(fontsize=8, title="target quantile")
    fig.suptitle("Quantile-indexed utility surface: information about $q_p$ from "
                 "stimulating at the level-$r$ point", fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# Phase 2: SBC and coverage
# --------------------------------------------------------------------------


def plot_sbc(raw: pd.DataFrame, targets, name="phase2_sbc"):
    policies = sorted(raw.policy.unique())
    fig, axes = plt.subplots(len(policies), len(targets),
                             figsize=(3.1 * len(targets), 3.0 * len(policies)),
                             squeeze=False)
    for i, policy in enumerate(policies):
        g = raw[raw.policy == policy]
        for j, t in enumerate(targets):
            ax = axes[i][j]
            n = len(g)
            band = 1.358 / np.sqrt(n)
            grid = np.linspace(0, 1, 200)
            ax.fill_between(grid, grid - band, grid + band, color="0.85",
                            label="95% band" if i == j == 0 else None)
            for backend, colour in (("ref", "C0"), ("particle", "C3")):
                col = f"{t}_rank_{backend}"
                if col not in g:
                    continue
                r = np.sort(g[col].to_numpy())
                ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=colour, lw=1.4,
                        label=backend if i == j == 0 else None)
            ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls="--")
            ax.set_title(f"{policy} / {t}", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Phase 2: SBC rank ECDFs with the adaptive policy inside the "
                 "simulator", fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


def plot_coverage_map(cov: pd.DataFrame, targets=("q0.5", "q0.95", "q0.99"),
                      backend="particle", name="phase2_coverage_map"):
    sel = cov[(cov.backend == backend) & (cov.target.isin(targets))]
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.6))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        piv = sel[sel.target == t].pivot(index="sigma_true", columns="mu_true",
                                         values="coverage")
        im = ax.imshow(piv.values, vmin=0.80, vmax=1.0, cmap="viridis",
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel(r"true $\mu$")
        ax.set_title(t, fontsize=10)
        se = sel[sel.target == t].pivot(index="sigma_true", columns="mu_true",
                                        values="coverage_se")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.3f}\n±{se.values[i, j]:.3f}",
                        ha="center", va="center", fontsize=7.5, color="w")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel(r"true $\sigma$")
    fig.suptitle(f"Phase 2: fixed-$\\theta$ 95% coverage ({backend} posterior); "
                 "nominal 0.95", fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# Phase 3 figures
# --------------------------------------------------------------------------

POLICY_LABEL_3 = {
    "entropy_vector": "adaptive MI $(\\mu,\\sigma)$",
    "entropy_q0.95": "adaptive MI $q_{0.95}$",
    "fixed_design": "non-adaptive fixed",
    "uniform_grid": "broad exploratory",
}


def plot_atlas(summary: pd.DataFrame, block="I", backend="ref", n=50,
               targets=("q0.95", "q0.99"), name=None):
    """Coverage, interval width and false certainty on one canvas.

    The three have to be read together: low coverage with a wide interval is
    honest ignorance, low coverage with a narrow interval is the failure the
    thesis is named after.  Plotting them separately invites reading the first
    as the second.
    """
    sel = summary[(summary.block == block) & (summary.backend == backend)
                  & (summary.n == n) & summary.target.isin(targets)]
    if sel.empty:
        return None
    dgps = list(dict.fromkeys(sel.dgp))
    pols = [p for p in POLICY_ORDER + list(POLICY_LABEL_3) if p in set(sel.pol)]
    fig, axes = plt.subplots(len(targets), 3, figsize=(16, 4.2 * len(targets)),
                             squeeze=False)
    for i, t in enumerate(targets):
        g = sel[sel.target == t]
        for j, (metric, lab, ref_line) in enumerate(
            (("coverage", "coverage of physical quantile", 0.95),
             ("mean_width", "mean 95% interval width", None),
             ("false_certainty", "false certainty: misses AND narrow", None)),
        ):
            ax = axes[i][j]
            piv = g.pivot_table(index="dgp", columns="pol", values=metric).reindex(dgps)
            xs = np.arange(len(piv))
            w = 0.8 / max(len(pols), 1)
            for k, p in enumerate(pols):
                if p not in piv:
                    continue
                ax.bar(xs + k * w, piv[p].to_numpy(), w * 0.9,
                       label=POLICY_LABEL_3.get(p, p) if i == 0 and j == 0 else None)
            if metric == "mean_width" and "w_star" in g:
                ws = g[g.n == n].w_star.dropna()
                if len(ws):
                    ax.axhline(float(ws.iloc[0]), color="k", ls=":", lw=1.2,
                               label="$w^*$" if i == 0 else None)
            if ref_line is not None:
                ax.axhline(ref_line, color="k", ls="--", lw=1.0)
            ax.set_xticks(xs + 0.4)
            ax.set_xticklabels(piv.index, rotation=40, ha="right", fontsize=7.5)
            ax.set_title(f"{t} — {lab}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7.5)
    fig.suptitle(f"Phase 3B failure atlas — block {block}, {backend} posterior, n = {n}",
                 fontsize=12)
    fig.tight_layout()
    return _save(fig, name or f"phase3b_atlas_block{block}_{backend}_n{n}")


def plot_nooverlap_gap(retained: pd.DataFrame, name="phase3a_nooverlap_gap"):
    """Particle-minus-reference behaviour on identical histories."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    arms = ["overlap_control", "no_overlap"]
    colours = {"overlap_control": "C0", "no_overlap": "C3"}

    ax = axes[0]
    rows, labels = [], []
    for mech in sorted(retained.mechanism.unique()):
        for arm in arms:
            g = retained[(retained.mechanism == mech) & (retained.arm == arm)]
            if len(g) < 10:
                continue
            for t in ("q0.5", "q0.95", "q0.99"):
                cd = g[f"{t}_cov_diff"].to_numpy()
                rows.append((cd.mean(), 1.96 * cd.std(ddof=1) / np.sqrt(len(cd)),
                             colours[arm]))
                labels.append(f"{mech[:12]}\n{arm[:10]}\n{t}")
    ys = np.arange(len(rows))
    ax.barh(ys, [r[0] for r in rows], xerr=[r[1] for r in rows],
            color=[r[2] for r in rows], height=0.7)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(-0.02, color="0.4", ls=":", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("particle − reference coverage (paired)")
    ax.set_title("Paired coverage difference", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    for arm in arms:
        g = retained[retained.arm == arm]
        if len(g) < 10:
            continue
        ax.scatter(g["particle_ess"], g["q0.95_rank_diff"], s=5, alpha=0.25,
                   color=colours[arm], label=arm)
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("particle ESS (log scale)")
    ax.set_ylabel("rank difference, $q_{0.95}$")
    ax.set_title("Degeneracy drives the gap", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for arm in arms:
        g = retained[(retained.arm == arm) & retained["q0.95_sd_ratio"].notna()]
        if len(g) < 10:
            continue
        ax.hist(g["q0.95_sd_ratio"], bins=40, range=(0.4, 1.6), alpha=0.55,
                color=colours[arm], label=arm, density=True)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("particle sd / reference sd, $q_{0.95}$")
    ax.set_title("Particle posterior is too narrow", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("Phase 3A: particle vs reference on identical no-overlap histories",
                 fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


def plot_design_distributions(designs: pd.DataFrame, cells, grid=None,
                              name="phase3b_design_distributions"):
    """Where each policy actually put its stimuli."""
    fig, axes = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 4),
                             squeeze=False, sharey=True)
    for ax, cell in zip(axes[0], cells):
        g = designs[designs.cell == cell]
        if g.empty:
            continue
        x = np.concatenate([np.asarray(v, dtype=float) for v in g["x"].tolist()])
        ax.hist(x, bins=60, density=True, color="C0", alpha=0.8)
        _, dgp, pol, _, _ = cell.split("|")
        ax.set_title(f"{dgp}\n{POLICY_LABEL_3.get(pol, pol)}", fontsize=9)
        ax.set_xlabel("stimulus")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("empirical design density")
    fig.suptitle("Phase 3B: empirical design distributions $\\nu_a$", fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


def plot_pseudo_truth(pseudo: pd.DataFrame, block="I", scale="s3", target="q0.95",
                      name=None):
    """Physical truth vs policy-dependent pseudo-true quantiles."""
    sel = pseudo[(pseudo.block == block) & (pseudo.scale == scale)]
    if sel.empty:
        return None
    dgps = list(dict.fromkeys(sel.dgp))
    pols = [p for p in POLICY_LABEL_3 if p in set(sel.pol)]
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(dgps))
    w = 0.8 / max(len(pols), 1)
    for k, p in enumerate(pols):
        g = sel[sel.pol == p].set_index("dgp").reindex(dgps)
        ax.errorbar(xs + k * w, g[f"pseudo_{target}"],
                    yerr=1.96 * g[f"pseudo_{target}_se"], fmt="o", ms=5,
                    capsize=2, label=POLICY_LABEL_3.get(p, p))
    phys = sel.groupby("dgp")[f"physical_{target}"].first().reindex(dgps)
    ax.plot(xs + 0.4, phys.to_numpy(), "k_", ms=28, mew=2,
            label="physical truth")
    ax.set_xticks(xs + 0.4)
    ax.set_xticklabels(dgps, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"{target} (stress units)")
    ax.set_title("Phase 3B: policy-dependent pseudo-true quantiles vs physical truth "
                 f"(block {block}, {target})", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, name or f"phase3b_pseudo_truth_{block}_{target}")
