"""Recover summary tables printed to stdout by the original (missing) pipeline
runs, from the committed results/*.log files.

This is NOT a sanctioned results/ writer (see CLAUDE.md #3 and #6): it runs no
experiment, mints no run_id, and cannot recover per-replicate raw data or MC
seeding. It only re-parses text that a runner already printed and that is
already committed to git (commit 4747866). Output goes to
docs/recovered_from_logs/, never to results/summaries/, so the provenance
contract is not violated by implication. Filenames match the original
manifest-referenced summary CSVs where the recovered table is a full,
faithful reconstruction; a table that is only a partial recovery (e.g. Block
I of phase3b_screen, with Block II absent from the log) is named to say so.

Run: python tools/reconstruct_from_logs.py
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results"
OUT = ROOT / "docs" / "recovered_from_logs"


def _write(df: pd.DataFrame, name: str) -> None:
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"wrote {path.relative_to(ROOT)}  ({len(df)} rows)")


def _table_block(lines: list[str], header_idx: int) -> list[str]:
    """Contiguous non-blank lines starting at header_idx."""
    block = []
    for line in lines[header_idx:]:
        if not line.strip():
            break
        block.append(line)
    return block


def phase1_utility() -> None:
    text = (LOGS / "phase1_utility.log").read_text()
    lines = text.splitlines()
    header = next(i for i, l in enumerate(lines) if l.strip().startswith("n "))
    block = "\n".join(_table_block(lines, header))
    df = pd.read_csv(io.StringIO(block), sep=r"\s+")
    _write(df, "phase1_utility_surface.csv")


def phase1_quadrature_audit() -> None:
    text = (LOGS / "phase1_quad_audit.log").read_text()
    lines = text.splitlines()

    header = next(i for i, l in enumerate(lines) if l.strip().startswith("target  c_fraction"))
    block = "\n".join(_table_block(lines, header))
    df = pd.read_csv(io.StringIO(block), sep=r"\s+")
    _write(df, "phase1_quadrature_audit_rotem_settings.csv")

    hits = [i for i, l in enumerate(lines) if l.strip() == "target             mu      sigma"]
    if hits:
        block = "\n".join(_table_block(lines, hits[0]))
        df = pd.read_csv(io.StringIO(block), sep=r"\s+")
        df = df.rename(columns={"target": "c_fraction"})
        _write(df, "phase1_quadrature_audit_bias_vs_bandwidth.csv")


def phase2_coverage_map() -> None:
    text = (LOGS / "phase2_coverage.log").read_text()
    lines = text.splitlines()
    header = next(i for i, l in enumerate(lines) if l.strip().startswith("mu_true"))
    end = next(i for i, l in enumerate(lines) if "manifest ->" in l)
    block = "\n".join(l for l in lines[header:end] if l.strip())
    df = pd.read_csv(io.StringIO(block), sep=r"\s+")
    _write(df, "phase2_coverage_map.csv")


def phase2_sbc() -> None:
    text = (LOGS / "phase2_sbc.log").read_text()
    lines = text.splitlines()
    header = next(i for i, l in enumerate(lines) if l.strip().startswith("policy"))
    end = next(i for i, l in enumerate(lines) if "manifest ->" in l)
    block = "\n".join(l for l in lines[header:end] if l.strip())
    df = pd.read_csv(io.StringIO(block), sep=r"\s+")
    _write(df, "phase2_sbc_uniformity.csv")


def phase3_reference_audit() -> None:
    text = (LOGS / "phase3_ref_audit.log").read_text()
    lines = text.splitlines()
    header = next(i for i, l in enumerate(lines) if l.strip().startswith("target  n"))
    block = "\n".join(_table_block(lines, header))
    df = pd.read_csv(io.StringIO(block), sep=r"\s+")
    _write(df, "phase3_reference_audit.csv")

    pass_line = next(l for l in lines if l.startswith("AUDIT PASSED"))
    m = re.search(
        r"max rank difference ([\d.eE+-]+) \(tolerance ([\d.]+)\), "
        r"coverage disagreements (\d+)/(\d+)",
        pass_line,
    )
    summary = pd.DataFrame([{
        "audit_passed": True,
        "max_rank_diff": float(m.group(1)),
        "tolerance": float(m.group(2)),
        "coverage_disagreements": int(m.group(3)),
        "coverage_disagreement_denominator": int(m.group(4)),
    }])
    _write(summary, "phase3_reference_audit_summary.csv")


def phase3a_pilot() -> None:
    text = (LOGS / "phase3a_pilot.log").read_text()
    lines = text.splitlines()
    header = next(i for i, l in enumerate(lines) if l.strip().startswith("mu_true"))
    # table is: header row of mu_true values, then 'sigma_true' index rows, blank line ends it
    block_lines = []
    for l in lines[header:]:
        if not l.strip():
            break
        block_lines.append(l)
    wide = pd.read_csv(io.StringIO("\n".join(block_lines)), sep=r"\s+", index_col=0)
    wide = wide.dropna(how="all")  # drops the blank "sigma_true" index-name line
    wide.index.name = "sigma_true"
    long = wide.reset_index().melt(id_vars="sigma_true", var_name="mu_true", value_name="no_overlap_rate")
    long["mu_true"] = long["mu_true"].astype(float)
    _write(long, "phase3a_pilot_rates.csv")


def phase3a_attribution() -> None:
    text = (LOGS / "phase3a_attribution.log").read_text()
    lines = text.splitlines()
    header = next(i for i, l in enumerate(lines) if l.strip().startswith("sigma_true"))
    block = _table_block(lines, header)
    cols = block[0].split()
    # 'arm' values contain spaces ("N=10k no-rejuv"), so split from the *right*
    # using the known trailing numeric-column count.
    n_trailing = len(cols) - 2  # everything after sigma_true, arm
    rows = []
    for l in block[1:]:
        parts = l.split()
        sigma_true = parts[0]
        trailing = parts[-n_trailing:]
        arm = " ".join(parts[1:-n_trailing])
        rows.append([sigma_true, arm, *trailing])
    df = pd.DataFrame(rows, columns=cols)
    numeric_cols = [c for c in cols if c not in ("arm",)]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c])
    _write(df, "phase3a_attribution.csv")


def phase3a_nooverlap() -> None:
    text = (LOGS / "phase3a_main.log").read_text()
    lines = text.splitlines()
    header = next(i for i, l in enumerate(lines) if l.strip().startswith("arm "))
    end = next(i for i, l in enumerate(lines) if "manifest ->" in l)
    block = "\n".join(l for l in lines[header:end] if l.strip())
    df = pd.read_csv(io.StringIO(block), sep=r"\s+")
    _write(df, "phase3a_nooverlap.csv")


def phase3b_confirm() -> None:
    text = (LOGS / "phase3b_confirm.log").read_text()
    lines = text.splitlines()

    header = next(i for i, l in enumerate(lines) if l.strip().startswith("label      dgp"))
    block = "\n".join(_table_block(lines, header))
    df = pd.read_csv(io.StringIO(block), sep=r"\s+")
    _write(df, "phase3b_confirm_summary.csv")

    header = next(i for i, l in enumerate(lines) if l.strip().startswith("label      dgp            pol  n_replicates"))
    block = "\n".join(_table_block(lines, header))
    df = pd.read_csv(io.StringIO(block), sep=r"\s+")
    _write(df, "phase3b_confirm_stopping.csv")

    header = next(i for i, l in enumerate(lines) if l.strip().startswith("target  n_pairs"))
    block = "\n".join(_table_block(lines, header))
    df = pd.read_csv(io.StringIO(block), sep=r"\s+")
    _write(df, "phase3b_confirm_h2_paired.csv")


def phase3b_pseudo() -> None:
    text = (LOGS / "phase3b_pseudo.log").read_text()
    lines = text.splitlines()
    header = next(i for i, l in enumerate(lines) if l.strip().startswith("dgp target"))
    end = next(i for i, l in enumerate(lines) if "manifest ->" in l)
    block = "\n".join(l for l in lines[header:end] if l.strip())
    df = pd.read_csv(io.StringIO(block), sep=r"\s+")
    _write(df, "phase3b_pseudo_truth_spread.csv")


def phase3b_screen() -> None:
    text = (LOGS / "phase3b_screen.log").read_text()
    lines = text.splitlines()

    def crosstab(start_marker: str, value_name: str) -> pd.DataFrame:
        start = next(i for i, l in enumerate(lines) if l.strip() == start_marker)
        pol_header = lines[start].strip()
        policies = pol_header.split()[1:]  # drop the leading "pol" column label
        rows = []
        dgp = None
        for l in lines[start + 2:]:
            if not l.strip():
                break
            parts = l.split()
            # rows are "<dgp> <target> v1 v2 v3 v4" or "<target> v1 v2 v3 v4"
            # (dgp blank/carried over for the second target row)
            n_vals = len(policies)
            vals = parts[-n_vals:]
            rest = parts[:-n_vals]
            if len(rest) == 2:
                dgp, target = rest
            else:
                target = rest[0]
            for pol, v in zip(policies, vals):
                rows.append({"dgp": dgp, "target": target, "policy": pol, value_name: float(v)})
        return pd.DataFrame(rows)

    fc = crosstab("pol                  entropy_q0.95  entropy_vector  fixed_design  uniform_grid", "false_certainty")
    # second crosstab has the same header text repeated for "coverage:"
    cov_start = next(i for i, l in enumerate(lines) if l.strip() == "coverage:")
    cov_lines = lines[cov_start:]

    def crosstab_from(lines_local, value_name):
        pol_header = lines_local[1].strip()
        policies = pol_header.split()[1:]  # drop the leading "pol" column label
        rows = []
        dgp = None
        for l in lines_local[3:]:
            if not l.strip() or "manifest ->" in l:
                break
            parts = l.split()
            n_vals = len(policies)
            vals = parts[-n_vals:]
            rest = parts[:-n_vals]
            if len(rest) == 2:
                dgp, target = rest
            else:
                target = rest[0]
            for pol, v in zip(policies, vals):
                rows.append({"dgp": dgp, "target": target, "policy": pol, value_name: float(v)})
        return pd.DataFrame(rows)

    cov = crosstab_from(cov_lines, "coverage")

    merged = fc.merge(cov, on=["dgp", "target", "policy"], how="outer")
    merged.insert(0, "block", "I")
    merged.insert(1, "n", 50)
    merged.insert(2, "posterior", "reference")
    _write(merged, "phase3b_screen_summary_blockI_reference.csv")
    print("NOTE: Block II tables are not present in phase3b_screen.log (log is "
          "truncated relative to the 88-cell run) — not recoverable from this "
          "source. See docs/discrepancies.md.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phase1_utility()
    phase1_quadrature_audit()
    phase2_coverage_map()
    phase2_sbc()
    phase3_reference_audit()
    phase3a_pilot()
    phase3a_attribution()
    phase3a_nooverlap()
    phase3b_confirm()
    phase3b_pseudo()
    phase3b_screen()


if __name__ == "__main__":
    main()
