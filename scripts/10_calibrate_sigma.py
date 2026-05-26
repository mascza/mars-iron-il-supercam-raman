"""Commit E.3 investigation: full-corpus SNR distribution and 5sigma calibration check.

Reads results/raw_scores.parquet and results/matches.parquet to characterize
the per-(compound x A-peak) snr_detrended distribution across all 1,436 Mars
observations. Compares the distribution tail at the production 5sigma
strong-tier threshold to:
  - lab self-match A-primary SNRs from 06_validate_self_match.py (5 reference
    points covering the 4 compounds, with Fe2Cl7 contributing two values)
  - hypothetical 6sigma and 7sigma threshold-tightening alternatives

The investigation question is whether the 5sigma threshold sits at a
defensible position in the per-peak SNR distribution, or whether
threshold-tightening to 6 or 7 sigma would improve specificity (reduce
strong_candidate count from residual-correlation noise) without sacrificing
recall on validator clean-lab cases. The output is a recommendation, not
an implementation; any actual threshold change is a separate commit
outside commit E proper.

Investigation pattern shifts from per-observation (E.1, E.2) to corpus-wide
distributional characterization. Same investigation discipline (kind='stable'
where applicable, defensive drop_duplicates, int64 obs-key dtypes,
Helvetica + pdf.fonttype=42, Wong palette), different scaffold (no
per-observation spectrum or zoomed panels; instead per-peak histograms +
CDFs in a single 3x4 grid).

Inputs:
  results/matches.parquet
  results/raw_scores.parquet

Outputs:
  docs/figures/e_investigations/sigma_calibration.png  (300 dpi)
  docs/figures/e_investigations/sigma_calibration.pdf  (vector)
  Numerical tables to stdout (4 tables plus reading-prompt summary).

Scope: commit E.3 of 5. Investigation only; no matcher, catalog, or
parquet changes. The figure directory is tracked (committed artifact;
established by E.1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Inputs.
MATCHES_PATH = REPO_ROOT / "results" / "matches.parquet"
RAW_SCORES_PATH = REPO_ROOT / "results" / "raw_scores.parquet"

# Outputs (tracked).
FIG_DIR = REPO_ROOT / "docs" / "figures" / "e_investigations"
FIG_PNG = FIG_DIR / "sigma_calibration.png"
FIG_PDF = FIG_DIR / "sigma_calibration.pdf"

COMPOUND_ORDER = (
    "EMIM-FeCl4",
    "EMIM-FeBr4",
    "EMIM2-Fe2Cl7",
    "EMIM2-FeSO4",
)

# Color palette: Wong color-blind-safe, no red-green pairing. Same as
# E.1, E.2.
COMPOUND_COLOR = {
    "EMIM-FeCl4": "#0072B2",     # blue
    "EMIM-FeBr4": "#E69F00",     # orange
    "EMIM2-Fe2Cl7": "#CC79A7",   # reddish-purple
    "EMIM2-FeSO4": "#009E73",    # bluish green
}

# All 11 catalog Class A peaks, in (compound, position, subclass) order.
# Verified during design review against peak_catalog.parquet:
#   EMIM-FeCl4    1 primary + 1 secondary = 2
#   EMIM-FeBr4    1 primary + 4 secondary = 5
#   EMIM2-Fe2Cl7  2 primary + 1 secondary = 3
#   EMIM2-FeSO4   1 primary + 0 secondary = 1
# Total 11. The v4 handoff said 13; that was a transcription error
# (Fe2Cl7 was claimed as 2P+4S but the catalog has 2P+1S).
A_PEAKS: list[tuple[str, float, str]] = [
    ("EMIM-FeCl4",   330.52, "primary"),
    ("EMIM-FeCl4",   386.31, "secondary"),
    ("EMIM-FeBr4",   218.40, "primary"),
    ("EMIM-FeBr4",   293.64, "secondary"),
    ("EMIM-FeBr4",   391.14, "secondary"),
    ("EMIM-FeBr4",   441.93, "secondary"),
    ("EMIM-FeBr4",   489.96, "secondary"),
    ("EMIM2-Fe2Cl7", 190.94, "primary"),
    ("EMIM2-Fe2Cl7", 326.10, "primary"),
    ("EMIM2-Fe2Cl7", 460.31, "secondary"),
    ("EMIM2-FeSO4",  958.72, "primary"),
]

# Lab self-match A-primary SNRs from scripts/06_validate_self_match.py.
# Verified fresh during E.3 design review by re-importing and re-running
# 06_validate_self_match.score_cell on each diagonal cell. The validator
# prints these in its stdout report; the values here are the per-A-primary
# SNRs, with Fe2Cl7 contributing two values (190.94 and 326.10) because
# it has two A-primary peaks. Citation: scripts/06_validate_self_match.py.
VALIDATOR_SELF_MATCH_SNR: dict[tuple[str, float], float] = {
    ("EMIM-FeCl4",   330.52):  7.32,
    ("EMIM-FeBr4",   218.40): 10.44,
    ("EMIM2-Fe2Cl7", 190.94):  3.06,
    ("EMIM2-Fe2Cl7", 326.10):  9.19,
    ("EMIM2-FeSO4",  958.72):  4.37,
}

# Threshold-tightening grid for the recall and strong_candidate-count
# analysis.
THRESHOLDS: tuple[float, ...] = (5.0, 6.0, 7.0)

# Distribution-shape diagnostic cutoff. p99/p50 > HEAVY_TAIL_RATIO_CUTOFF
# indicates heavy-tailed (consistent with real signal mixed with noise);
# below indicates piled near mode (consistent with residual-correlation
# noise reaching the gate). Cutoff value is interpretive; per design
# review it is set at 3.0 with the expectation that all 11 peaks will
# clear it.
HEAVY_TAIL_RATIO_CUTOFF = 3.0

# Strong-tier gate constants, mirrored from 06/07.
COSINE_TOP_PERCENTILE = 90.0
THRESH_SECONDARY = 3.0

# Plot styling for the threshold and validator verticals.
THRESHOLD_STYLE = {
    5.0: dict(color="#D55E00", lw=1.3, ls="-"),
    6.0: dict(color="#D55E00", lw=1.1, ls="--"),
    7.0: dict(color="#D55E00", lw=0.9, ls=":"),
}
VALIDATOR_STYLE = dict(color="#000000", lw=1.5, ls=(0, (4, 2)))

# Histogram x-axis range. Bulk of every per-peak distribution falls inside
# this range; tails extending past X_HI are annotated per-panel as text
# rather than plotted, to keep the threshold-line region at consistent
# resolution across panels.
X_LO, X_HI = -5.0, 25.0


def short_label(compound: str) -> str:
    if compound.startswith("EMIM2-"):
        return compound[6:]
    if compound.startswith("EMIM-"):
        return compound[5:]
    return compound


def load_raw_scores() -> pd.DataFrame:
    df = pd.read_parquet(RAW_SCORES_PATH)
    for col in ("sol", "sclk_int", "point_number"):
        df[col] = df[col].astype("int64")
    return df


def load_matches() -> pd.DataFrame:
    df = pd.read_parquet(MATCHES_PATH)
    for col in ("sol", "sclk_int", "point_number"):
        df[col] = df[col].astype("int64")
    return df


def per_peak_snr_arrays(
    raw_scores: pd.DataFrame,
) -> dict[tuple[str, float], np.ndarray]:
    """Return {(compound, center): snr_array} for each of the 11 A peaks.
    Each array has 1,436 values (one per Mars observation).
    """
    out: dict[tuple[str, float], np.ndarray] = {}
    for compound, ctr, _sub in A_PEAKS:
        sub = raw_scores[
            (raw_scores["compound"] == compound)
            & (np.abs(raw_scores["peak_center_catalog"] - ctr) < 0.05)
        ]
        out[(compound, ctr)] = sub["snr_detrended"].to_numpy(dtype=float)
    return out


def per_obs_max_a_primary_snr(raw_scores: pd.DataFrame) -> pd.DataFrame:
    """For each (sol, sclk_int, seqid, point_number, compound), compute
    max snr_detrended across A-primary peaks. Returns one row per
    (obs, compound). The matcher's gate uses max-of-primaries semantics
    (n_a_primary_above_X >= 1), so this column is the operative quantity
    for threshold-tightening.
    """
    keys = ["sol", "sclk_int", "seqid", "point_number", "compound"]
    sub = raw_scores[
        (raw_scores["peak_class"] == "A")
        & (raw_scores["peak_subclass"] == "primary")
    ].copy()
    grp = sub.groupby(keys, sort=False)["snr_detrended"].max()
    return grp.rename("max_a_primary_snr").reset_index()


def print_table_1(
    snr_arrays: dict[tuple[str, float], np.ndarray],
) -> None:
    """Per-peak distribution stats: 11 rows, one per A peak."""
    print("=" * 78)
    print("Table 1 - Per-peak SNR distribution stats (n=1436 per peak)")
    print("=" * 78)
    rows = []
    for compound, ctr, sub in A_PEAKS:
        arr = snr_arrays[(compound, ctr)]
        n = len(arr)
        sm = VALIDATOR_SELF_MATCH_SNR.get((compound, ctr), float("nan"))
        p50 = float(np.percentile(arr, 50))
        p99 = float(np.percentile(arr, 99))
        ratio = (p99 / p50) if abs(p50) > 1e-6 else float("inf")
        rows.append({
            "compound": short_label(compound),
            "pos": float(ctr),
            "sub": sub,
            "n": int(n),
            "min": float(np.min(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": p50,
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": p99,
            "max": float(np.max(arr)),
            "p99/p50": ratio,
            "self_match_snr": sm,
            "%>=5sigma": float((arr >= 5.0).sum()) / n * 100.0,
            "%>=6sigma": float((arr >= 6.0).sum()) / n * 100.0,
            "%>=7sigma": float((arr >= 7.0).sum()) / n * 100.0,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()


def print_table_2(raw_scores: pd.DataFrame) -> None:
    """Per-compound aggregate counts: at each threshold, how many distinct
    (obs x compound) cells have max(A-primary SNR) >= threshold? This is
    the matcher's actual A-primary gate (max-of-primaries semantics).
    Sanity reconciliation against Table 1's per-peak %>=Xsigma columns.
    """
    print("=" * 78)
    print(
        "Table 2 - Per-compound aggregate: obs with max(A-primary SNR) >= threshold\n"
        "          (sanity reconciliation; matcher uses max-of-primaries)"
    )
    print("=" * 78)
    max_df = per_obs_max_a_primary_snr(raw_scores)
    rows = []
    for compound in COMPOUND_ORDER:
        sub = max_df[max_df["compound"] == compound]
        n = len(sub)
        row = {
            "compound": short_label(compound),
            "n_obs": int(n),
        }
        for thresh in THRESHOLDS:
            n_above = int((sub["max_a_primary_snr"] >= thresh).sum())
            row[f"n>={thresh}sigma"] = n_above
            row[f"%>={thresh}sigma"] = (
                float(n_above) / n * 100.0 if n else float("nan")
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()


def print_table_3(
    raw_scores: pd.DataFrame, matches: pd.DataFrame,
) -> None:
    """Threshold-tightening: hypothetical strong_candidate counts at
    5sigma / 6sigma / 7sigma A-primary thresholds (secondary >= 3sigma
    and cosine >= 90th percentile gates unchanged). Plus per-compound
    validator-recall margin column.

    FeSO4 is structurally locked out of strong_candidate at any threshold
    because the catalog has 0 A-secondaries, so the secondary 3sigma gate
    is unsatisfiable. Annotated explicitly in the structural_lockout
    column. This is threshold-independent and is NOT a recall failure of
    the threshold-tightening exercise.
    """
    print("=" * 78)
    print(
        "Table 3 - Threshold-tightening: hypothetical strong_candidate counts\n"
        "          + validator self-match recall margin per compound"
    )
    print("=" * 78)
    keys = ["sol", "sclk_int", "seqid", "point_number", "compound"]
    max_df = per_obs_max_a_primary_snr(raw_scores)

    per_compound_self_match_max: dict[str, float] = {}
    for compound in COMPOUND_ORDER:
        primaries = [
            VALIDATOR_SELF_MATCH_SNR[(c, p)]
            for (c, p, sub) in A_PEAKS
            if c == compound and sub == "primary"
            and (c, p) in VALIDATOR_SELF_MATCH_SNR
        ]
        per_compound_self_match_max[compound] = (
            max(primaries) if primaries else float("nan")
        )

    a_secondary_counts: dict[str, int] = {}
    for compound in COMPOUND_ORDER:
        a_secondary_counts[compound] = sum(
            1 for (c, _p, sub) in A_PEAKS
            if c == compound and sub == "secondary"
        )

    rows = []
    total_per_thresh: dict[float, int] = {t: 0 for t in THRESHOLDS}
    for compound in COMPOUND_ORDER:
        sm = per_compound_self_match_max[compound]
        struct_lock = a_secondary_counts[compound] == 0
        row: dict = {
            "compound": short_label(compound),
            "max_self_match_snr": sm,
            "structural_lockout": "yes" if struct_lock else "no",
        }
        max_compound = max_df[max_df["compound"] == compound]
        match_compound = matches[matches["compound"] == compound][
            keys + [
                "n_a_secondary_above_3sigma",
                "cosine_percentile_within_compound_full_corpus",
            ]
        ]
        joined = match_compound.merge(
            max_compound, on=keys, how="left"
        )
        joined["max_a_primary_snr"] = joined["max_a_primary_snr"].fillna(0.0)
        for thresh in THRESHOLDS:
            n_strong = int(
                (
                    (joined["max_a_primary_snr"] >= thresh)
                    & (joined["n_a_secondary_above_3sigma"] >= 1)
                    & (joined["cosine_percentile_within_compound_full_corpus"]
                       >= COSINE_TOP_PERCENTILE)
                ).sum()
            )
            row[f"n_strong@{thresh}"] = n_strong
            margin = sm - thresh if not np.isnan(sm) else float("nan")
            row[f"margin@{thresh}"] = margin
            if struct_lock:
                recall = "lockout"
            elif np.isnan(sm):
                recall = "n/a"
            elif sm >= thresh:
                recall = "PASS"
            else:
                recall = "FAIL"
            row[f"recall@{thresh}"] = recall
            total_per_thresh[thresh] += n_strong
        rows.append(row)
    df = pd.DataFrame(rows)
    cols_order = ["compound", "max_self_match_snr", "structural_lockout"]
    for thresh in THRESHOLDS:
        cols_order += [
            f"n_strong@{thresh}",
            f"margin@{thresh}",
            f"recall@{thresh}",
        ]
    print(df[cols_order].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()
    print(
        "Total strong_candidate counts across all compounds: "
        + ", ".join(
            f"{t}sigma={n}" for t, n in total_per_thresh.items()
        )
    )
    print()
    print(
        "Detail rows below: per-A-primary self-match SNR + drives_gate flag.\n"
        "Gate uses max-of-primaries (already applied in the recall column\n"
        "above); rows with drives_gate=no are reported for completeness and\n"
        "connect to E.1 / E.2 findings that the chemically-distinctive\n"
        "primary peak (Fe2Cl7 190.94 Fe-Cl-Fe bridge) does not contribute\n"
        "to gate evaluation regardless of threshold."
    )
    print()
    detail_rows = []
    for compound, ctr, sub in A_PEAKS:
        if sub != "primary":
            continue
        sm = VALIDATOR_SELF_MATCH_SNR.get((compound, ctr), float("nan"))
        sm_max = per_compound_self_match_max[compound]
        is_max = (
            "yes" if (not np.isnan(sm)) and abs(sm - sm_max) < 1e-6
            else "no"
        )
        drives = (
            "yes" if is_max == "yes"
            else "no (max-of-primaries; this peak silent does not change verdict)"
        )
        detail_rows.append({
            "compound": short_label(compound),
            "primary_pos": float(ctr),
            "self_match_snr": sm,
            "is_compound_max": is_max,
            "drives_gate": drives,
        })
    print(
        pd.DataFrame(detail_rows).to_string(
            index=False, float_format=lambda v: f"{v:.2f}"
        )
    )
    print()


def print_table_4(
    snr_arrays: dict[tuple[str, float], np.ndarray],
) -> None:
    """Distribution-shape diagnostic per peak: p99/p50 ratio. Above the
    HEAVY_TAIL_RATIO_CUTOFF flag indicates heavy-tailed; below indicates
    piled near mode.
    """
    print("=" * 78)
    print(
        "Table 4 - Distribution-shape diagnostic per peak (p99/p50 ratio)\n"
        f"          cutoff at {HEAVY_TAIL_RATIO_CUTOFF}: above = heavy-tailed; "
        f"below = piled"
    )
    print("=" * 78)
    rows = []
    for compound, ctr, sub in A_PEAKS:
        arr = snr_arrays[(compound, ctr)]
        p50 = float(np.percentile(arr, 50))
        p99 = float(np.percentile(arr, 99))
        ratio = (p99 / p50) if abs(p50) > 1e-6 else float("inf")
        flag = "heavy" if ratio > HEAVY_TAIL_RATIO_CUTOFF else "piled"
        rows.append({
            "compound": short_label(compound),
            "pos": float(ctr),
            "sub": sub,
            "p50": p50,
            "p99": p99,
            "p99/p50": ratio,
            "shape": flag,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    n_heavy = int((df["shape"] == "heavy").sum())
    print()
    print(f"Heavy-tailed peaks: {n_heavy} of {len(df)}")
    print()


def report_reading_prompt(
    raw_scores: pd.DataFrame, matches: pd.DataFrame,
    snr_arrays: dict[tuple[str, float], np.ndarray],
) -> None:
    """Compact summary for the chat-side reading: heavy-tailed verdict
    and the per-compound margin at each threshold.
    """
    print("=" * 78)
    print(
        "Reading prompt - 5sigma threshold calibration vs lab self-match recall"
    )
    print("=" * 78)
    n_heavy = sum(
        1 for (c, p, _) in A_PEAKS
        if (
            np.percentile(snr_arrays[(c, p)], 99)
            / max(np.percentile(snr_arrays[(c, p)], 50), 1e-6)
            > HEAVY_TAIL_RATIO_CUTOFF
        )
    )
    print(
        f"  Distribution shape: {n_heavy} of {len(A_PEAKS)} A-peaks heavy-tailed "
        f"(p99/p50 > {HEAVY_TAIL_RATIO_CUTOFF}). Distributions are NOT piled "
        f"at threshold; rules out reading (c)."
    )
    print()
    keys = ["sol", "sclk_int", "seqid", "point_number", "compound"]
    max_df = per_obs_max_a_primary_snr(raw_scores)
    print("  Strong_candidate counts and per-compound recall margins:")
    for thresh in THRESHOLDS:
        total = 0
        margin_parts: list[str] = []
        for compound in COMPOUND_ORDER:
            primaries = [
                VALIDATOR_SELF_MATCH_SNR[(c, p)]
                for (c, p, sub) in A_PEAKS
                if c == compound and sub == "primary"
                and (c, p) in VALIDATOR_SELF_MATCH_SNR
            ]
            sm = max(primaries) if primaries else float("nan")
            margin = sm - thresh
            margin_parts.append(f"{short_label(compound)}={margin:+.2f}")
            max_compound = max_df[max_df["compound"] == compound]
            match_compound = matches[matches["compound"] == compound][
                keys + [
                    "n_a_secondary_above_3sigma",
                    "cosine_percentile_within_compound_full_corpus",
                ]
            ]
            joined = match_compound.merge(max_compound, on=keys, how="left")
            joined["max_a_primary_snr"] = joined["max_a_primary_snr"].fillna(0.0)
            n_strong = int(
                (
                    (joined["max_a_primary_snr"] >= thresh)
                    & (joined["n_a_secondary_above_3sigma"] >= 1)
                    & (joined["cosine_percentile_within_compound_full_corpus"]
                       >= COSINE_TOP_PERCENTILE)
                ).sum()
            )
            total += n_strong
        print(
            f"    {thresh}sigma:  total strong={total}  "
            f"margins ({', '.join(margin_parts)})"
        )
    print()


def make_plot(
    snr_arrays: dict[tuple[str, float], np.ndarray],
):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig = plt.figure(figsize=(14.0, 10.0))
    gs = fig.add_gridspec(
        nrows=3, ncols=4,
        left=0.06, right=0.95, top=0.92, bottom=0.06,
        hspace=0.55, wspace=0.45,
    )

    BIN_EDGES = np.linspace(X_LO, X_HI, 41)

    for idx, (compound, ctr, sub) in enumerate(A_PEAKS):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col])
        arr = snr_arrays[(compound, ctr)]
        n = len(arr)
        color = COMPOUND_COLOR[compound]

        ax.hist(
            arr, bins=BIN_EDGES, color=color, alpha=0.65,
        )
        ax.set_xlim(X_LO, X_HI)
        ax.set_xlabel("snr_detrended", fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

        ax2 = ax.twinx()
        sorted_arr = np.sort(arr)
        cdf_y = np.arange(1, n + 1) / n
        ax2.step(
            sorted_arr, cdf_y, where="post",
            color="black", lw=0.8,
        )
        ax2.set_ylim(0.0, 1.05)
        ax2.set_ylabel("CDF", fontsize=9)
        ax2.tick_params(axis="y", labelsize=8)
        ax2.axhline(0.95, color="black", lw=0.4, ls=":", alpha=0.4)
        ax2.axhline(0.99, color="black", lw=0.4, ls=":", alpha=0.4)

        for thresh, style in THRESHOLD_STYLE.items():
            ax.axvline(thresh, **style)

        sm = VALIDATOR_SELF_MATCH_SNR.get((compound, ctr), None)
        if sm is not None:
            ax.axvline(sm, **VALIDATOR_STYLE)

        p50 = float(np.percentile(arr, 50))
        p99 = float(np.percentile(arr, 99))
        ratio = (p99 / p50) if abs(p50) > 1e-6 else float("inf")
        pct5 = float((arr >= 5.0).sum()) / n * 100.0
        ax.set_title(
            f"{short_label(compound)} {ctr:.2f} ({sub})", fontsize=9,
        )
        annot = (
            f"n={n}  pct>=5sigma={pct5:.1f}%\n"
            f"p99/p50={ratio:.1f}  "
            + (f"self_match={sm:.2f}" if sm is not None else "self_match=n/a")
        )
        ax.text(
            0.97, 0.97, annot,
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        )

        # Annotate corpus max if it falls outside the panel x-range. The
        # bulk of the distribution and the threshold lines stay at the
        # configured resolution; the tail value is visible without
        # extending the x-range across panels.
        arr_max = float(np.max(arr))
        if arr_max > X_HI:
            ax.text(
                0.97, 0.78,
                f"max={arr_max:.2f} ->",
                transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="gray", alpha=0.85),
            )

    ax_legend = fig.add_subplot(gs[2, 3])
    ax_legend.axis("off")
    handles = [
        plt.Line2D([0], [0],
                   color=THRESHOLD_STYLE[5.0]["color"],
                   lw=THRESHOLD_STYLE[5.0]["lw"],
                   ls=THRESHOLD_STYLE[5.0]["ls"],
                   label="5sigma threshold (production)"),
        plt.Line2D([0], [0],
                   color=THRESHOLD_STYLE[6.0]["color"],
                   lw=THRESHOLD_STYLE[6.0]["lw"],
                   ls=THRESHOLD_STYLE[6.0]["ls"],
                   label="6sigma threshold (hypothetical)"),
        plt.Line2D([0], [0],
                   color=THRESHOLD_STYLE[7.0]["color"],
                   lw=THRESHOLD_STYLE[7.0]["lw"],
                   ls=THRESHOLD_STYLE[7.0]["ls"],
                   label="7sigma threshold (hypothetical)"),
        plt.Line2D([0], [0],
                   color=VALIDATOR_STYLE["color"],
                   lw=VALIDATOR_STYLE["lw"],
                   ls=VALIDATOR_STYLE["ls"],
                   label="lab self-match SNR (validator)"),
        plt.Line2D([0], [0], color="black", lw=0.8,
                   label="CDF (right y axis)"),
    ]
    ax_legend.legend(
        handles=handles, loc="center", fontsize=8,
        frameon=True, framealpha=0.9,
    )
    ax_legend.set_title(
        "Threshold and reference verticals", fontsize=9, pad=10,
    )

    fig.suptitle(
        "Commit E.3: Per-peak SNR distributions and 5sigma threshold calibration",
        fontsize=11, y=0.965,
    )
    return fig


def main() -> int:
    for p in (MATCHES_PATH, RAW_SCORES_PATH):
        if not p.is_file():
            raise FileNotFoundError(f"Missing input: {p}")

    # Fail-fast: every A-primary in A_PEAKS must have a validator entry.
    # Cheap insurance against future catalog changes silently misreporting
    # recall numbers in Table 3 / reading prompt.
    missing = [
        (c, p) for (c, p, sub) in A_PEAKS
        if sub == "primary" and (c, p) not in VALIDATOR_SELF_MATCH_SNR
    ]
    if missing:
        raise ValueError(
            f"A-primaries missing from VALIDATOR_SELF_MATCH_SNR: {missing}"
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw_scores = load_raw_scores()
    matches = load_matches()
    snr_arrays = per_peak_snr_arrays(raw_scores)

    expected_n = matches[
        ["sol", "sclk_int", "seqid", "point_number"]
    ].drop_duplicates(
        subset=["sol", "sclk_int", "seqid", "point_number"]
    ).shape[0]
    for key, arr in snr_arrays.items():
        if len(arr) != expected_n:
            raise ValueError(
                f"peak {key} has {len(arr)} rows, expected {expected_n}"
            )

    print_table_1(snr_arrays)
    print_table_2(raw_scores)
    print_table_3(raw_scores, matches)
    print_table_4(snr_arrays)
    report_reading_prompt(raw_scores, matches, snr_arrays)

    fig = make_plot(snr_arrays)
    fig.savefig(FIG_PNG, dpi=300)
    fig.savefig(FIG_PDF)
    plt.close(fig)
    print(f"Wrote {FIG_PNG} ({FIG_PNG.stat().st_size} bytes)")
    print(f"Wrote {FIG_PDF} ({FIG_PDF.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
