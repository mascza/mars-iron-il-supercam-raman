"""Commit E.4 investigation: position_offset_normalized grid-quantization floor.

Reads results/raw_scores.parquet to characterize the per-(compound x A-peak)
position_offset_normalized distribution across all 1,436 Mars observations.
The metric is computed in scripts/_processing.py compute_peak_quality_flags
as |peak_max_position - peak_center_catalog| / tolerance_used. Because
peak_max_position is the integer-valued argmax over the Mars 1.0 cm-1
grid within the tolerance window and peak_center_catalog is fractional,
the metric takes only a finite set of discrete values per peak.

The investigation question: where does position_offset_normalized carry
chemistry information beyond the per-peak quantization grid, and where
is it dominated by quantization?

Pattern follows E.3's corpus-distributional shape (3x4 panel grid,
per-peak analysis) but the question is metric-floor characterization,
not threshold tightening. There is no gate on position_offset_normalized
in 06; the metric drives only the n_quality_flags annotation in 07.
The output is documentation-only -- a manuscript-framing rule for how
to interpret position_offset values in candidate observations, plus a
discrete-not-coarse reframing of the metric. No 06 change is motivated.

Inputs:
  results/raw_scores.parquet (peak_center_catalog, peak_max_position,
    tolerance_used, position_offset_normalized are all in raw_scores;
    no other parquet needed)

Outputs:
  docs/figures/e_investigations/position_offset_quantization.png  (300 dpi)
  docs/figures/e_investigations/position_offset_quantization.pdf  (vector)
  Numerical tables to stdout (4 tables plus reading-prompt summary).

Scope: commit E.4 of 5. Investigation only; no matcher, catalog, or
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
RAW_SCORES_PATH = REPO_ROOT / "results" / "raw_scores.parquet"

# Outputs (tracked).
FIG_DIR = REPO_ROOT / "docs" / "figures" / "e_investigations"
FIG_PNG = FIG_DIR / "position_offset_quantization.png"
FIG_PDF = FIG_DIR / "position_offset_quantization.pdf"

COMPOUND_ORDER = (
    "EMIM-FeCl4",
    "EMIM-FeBr4",
    "EMIM2-Fe2Cl7",
    "EMIM2-FeSO4",
)

# Color palette: Wong color-blind-safe, no red-green pairing. Same as
# E.1, E.2, E.3.
COMPOUND_COLOR = {
    "EMIM-FeCl4": "#0072B2",     # blue
    "EMIM-FeBr4": "#E69F00",     # orange
    "EMIM2-Fe2Cl7": "#CC79A7",   # reddish-purple
    "EMIM2-FeSO4": "#009E73",    # bluish green
}

# All 11 catalog Class A peaks, in (compound, position-rounded, subclass)
# order. Verified against peak_catalog.parquet during E.3 design review.
# This script reads the EXACT catalog center per peak from
# raw_scores.peak_center_catalog rather than using the rounded values
# below, so the per-peak quantization grid is computed from full catalog
# precision.
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

# E.1 and E.2 documented cases for Table 4 revisitation. Includes the
# correction noted during design review: sol 996 Fe2Cl7 326.10 is 0.3805
# (k=328), not 0.81 as the user's E.4 prompt cited; 0.81 was the value
# for sol 996 Fe2Cl7 190.94. Pinned in this script and in the session
# log per append-only discipline.
SPECIFIC_CASES: list[tuple[int, int, str, int, str, float, str]] = [
    (996, 755369654, "scam05996", 32, "EMIM-FeCl4",   330.52, "E.1 sol 996 FeCl4 330.52"),
    (996, 755369654, "scam05996", 32, "EMIM2-Fe2Cl7", 326.10, "E.1 sol 996 Fe2Cl7 326.10"),
    (996, 755369654, "scam05996", 32, "EMIM2-Fe2Cl7", 190.94, "E.1 sol 996 Fe2Cl7 190.94"),
    (162, 681322174, "scam02162",  1, "EMIM-FeBr4",   218.40, "E.2 sol 162 FeBr4 218.40"),
    (162, 681322174, "scam02162",  1, "EMIM-FeBr4",   293.64, "E.2 sol 162 FeBr4 293.64"),
    (162, 681322174, "scam02162",  1, "EMIM-FeBr4",   489.96, "E.2 sol 162 FeBr4 489.96"),
]

# The matcher's de-facto position_offset quality-flag threshold (used by
# 07's aggregate_quality_flags via the position_offset_normalized > 0.8
# clause). Reference vertical for the per-panel plots.
POSITION_OFFSET_QUALITY_FLAG_THRESHOLD = 0.8

# Cutoff for the doublet-pair vs single-near-zero grid-pattern
# classification. If the gap between the two smallest grid values is
# less than this threshold, the grid has a doublet-pair structure (two
# integers roughly equidistant from the catalog center); otherwise the
# grid has a single-near-zero structure (one integer much closer than
# its neighbors).
DOUBLET_PAIR_GAP_CUTOFF = 0.1

# Plot styling reference verticals.
QFLAG_STYLE = dict(color="#D55E00", lw=1.2, ls="--")
MIN_OFFSET_STYLE = dict(color="#000000", lw=1.2, ls=":")


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


def exact_center_and_tolerance(
    raw_scores: pd.DataFrame, compound: str, rounded_pos: float,
) -> tuple[float, float]:
    """Read the exact peak_center_catalog and tolerance_used for one A-peak
    from raw_scores. Both are stored per-peak in the matcher's output;
    returning the values from raw_scores avoids any drift between the
    rounded values in A_PEAKS and the catalog values the matcher used.
    """
    sub = raw_scores[
        (raw_scores["compound"] == compound)
        & (np.abs(raw_scores["peak_center_catalog"] - rounded_pos) < 0.05)
    ]
    if sub.empty:
        raise RuntimeError(
            f"no raw_scores rows for ({compound}, {rounded_pos})"
        )
    centers = sub["peak_center_catalog"].unique()
    tols = sub["tolerance_used"].unique()
    if len(centers) != 1 or len(tols) != 1:
        raise RuntimeError(
            f"non-unique center or tolerance for ({compound}, "
            f"{rounded_pos}): centers={centers} tolerances={tols}"
        )
    return float(centers[0]), float(tols[0])


def quantization_grid(center: float, tolerance: float) -> list[float]:
    """Return sorted unique normalized-offset grid values for a peak with
    given exact catalog center and tolerance. Each grid value corresponds
    to an integer peak_max_position in [ceil(center - tol), floor(center
    + tol)].
    """
    k_lo = int(np.ceil(center - tolerance))
    k_hi = int(np.floor(center + tolerance))
    offsets = [
        abs(float(k) - center) / tolerance for k in range(k_lo, k_hi + 1)
    ]
    return sorted(set(round(o, 8) for o in offsets))


def grid_pattern(grid_values: list[float]) -> str:
    """Classify the grid as doublet-pair or single-near-zero based on the
    gap between the two smallest grid values. Doublet-pair = catalog
    center sits roughly halfway between two integers; single-near-zero =
    catalog center sits roughly at an integer.
    """
    nz = [v for v in grid_values if v > 1e-9]
    if len(nz) < 2:
        return "n/a"
    return (
        "doublet-pair"
        if (nz[1] - nz[0]) < DOUBLET_PAIR_GAP_CUTOFF
        else "single-near-zero"
    )


def per_peak_arrays(
    raw_scores: pd.DataFrame,
) -> dict[tuple[str, float], np.ndarray]:
    """Return {(compound, rounded_pos): position_offset_normalized array}
    for each of the 11 A peaks. Each array has 1,436 values.
    """
    out: dict[tuple[str, float], np.ndarray] = {}
    for compound, ctr, _sub in A_PEAKS:
        sub = raw_scores[
            (raw_scores["compound"] == compound)
            & (np.abs(raw_scores["peak_center_catalog"] - ctr) < 0.05)
        ]
        out[(compound, ctr)] = sub["position_offset_normalized"].to_numpy(
            dtype=float
        )
    return out


def empirical_count_at_grid(
    arr: np.ndarray, grid: list[float], tol: float = 1e-6,
) -> dict[float, int]:
    """Count empirical observations at each grid value (within tol).
    Returns {grid_value: count}.
    """
    counts: dict[float, int] = {g: 0 for g in grid}
    for value in arr:
        nearest = min(grid, key=lambda g: abs(g - value))
        if abs(nearest - value) <= tol:
            counts[nearest] += 1
    return counts


def print_table_1(
    raw_scores: pd.DataFrame,
) -> dict[tuple[str, float], dict]:
    """Per-peak quantization grid description: 11 rows, one per A-peak."""
    print("=" * 78)
    print(
        "Table 1 - Per-peak quantization grid (exact catalog centers from raw_scores)"
    )
    print("=" * 78)
    grid_meta: dict[tuple[str, float], dict] = {}
    rows = []
    for compound, ctr, sub in A_PEAKS:
        center, tolerance = exact_center_and_tolerance(
            raw_scores, compound, ctr
        )
        grid = quantization_grid(center, tolerance)
        nz = [v for v in grid if v > 1e-9]
        diffs = list(np.diff(grid)) if len(grid) >= 2 else [0.0]
        pattern = grid_pattern(grid)
        meta = {
            "exact_center": center,
            "tolerance": tolerance,
            "grid": grid,
            "min_non_zero": float(min(nz)) if nz else float("nan"),
            "n_grid": len(grid),
            "median_spacing": float(np.median(diffs)),
            "max_spacing": float(np.max(diffs)),
            "pattern": pattern,
        }
        grid_meta[(compound, ctr)] = meta
        rows.append({
            "compound": short_label(compound),
            "peak": float(ctr),
            "sub": sub,
            "exact_center": center,
            "tol": tolerance,
            "n_grid": len(grid),
            "min_non_zero": meta["min_non_zero"],
            "median_spacing": meta["median_spacing"],
            "max_spacing": meta["max_spacing"],
            "pattern": pattern,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(
        "Full grid values per peak (sorted ascending; each value corresponds"
    )
    print("to a specific integer peak_max_position within tolerance):")
    for compound, ctr, _sub in A_PEAKS:
        m = grid_meta[(compound, ctr)]
        g_str = ", ".join(f"{v:.4f}" for v in m["grid"])
        print(f"  {short_label(compound):<7} {ctr:>7.2f}  [{g_str}]")
    print()
    return grid_meta


def print_table_2(
    arrays: dict[tuple[str, float], np.ndarray],
    grid_meta: dict[tuple[str, float], dict],
) -> None:
    """Per-peak empirical distribution: 11 rows."""
    print("=" * 78)
    print(
        "Table 2 - Per-peak empirical distribution of position_offset_normalized"
    )
    print("=" * 78)
    rows = []
    for compound, ctr, sub in A_PEAKS:
        arr = arrays[(compound, ctr)]
        n = len(arr)
        m = grid_meta[(compound, ctr)]
        grid = m["grid"]
        min_nz = m["min_non_zero"]
        counts = empirical_count_at_grid(arr, grid)
        n_at_min = counts.get(min_nz, 0)
        n_above_qflag = int(
            (arr > POSITION_OFFSET_QUALITY_FLAG_THRESHOLD).sum()
        )
        n_on_grid = sum(counts.values())
        rows.append({
            "compound": short_label(compound),
            "peak": float(ctr),
            "sub": sub,
            "n": int(n),
            "min_observed": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
            "pct_on_grid": float(n_on_grid) / n * 100.0,
            "pct_at_min_offset": float(n_at_min) / n * 100.0,
            "pct_above_0.8": float(n_above_qflag) / n * 100.0,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()


def print_table_3(
    grid_meta: dict[tuple[str, float], dict],
    arrays: dict[tuple[str, float], np.ndarray],
) -> None:
    """Per-peak metric-resolution classification.

    Reading: position_offset_normalized is a fundamentally discrete
    metric with exactly 10 possible values per peak. Each value
    corresponds to a specific integer peak_max_position. The grid
    pattern (single-near-zero or doublet-pair) describes whether the
    closest-integer offset has one or two near-zero grid points, but in
    either case the metric is discrete-not-continuous and the manuscript
    implication is the same: report peak_max_position as integer
    wavenumber (or a discrete category) directly. The continuous
    interpretation of the normalized offset is misleading.
    """
    print("=" * 78)
    print(
        "Table 3 - Per-peak metric-resolution classification (discrete-not-coarse)\n"
        "          all 11 A-peaks have exactly 10 possible values"
    )
    print("=" * 78)
    rows = []
    for compound, ctr, sub in A_PEAKS:
        m = grid_meta[(compound, ctr)]
        rows.append({
            "compound": short_label(compound),
            "peak": float(ctr),
            "sub": sub,
            "tolerance": m["tolerance"],
            "n_grid_values": m["n_grid"],
            "median_spacing": m["median_spacing"],
            "max_spacing": m["max_spacing"],
            "pattern": m["pattern"],
            "interpretation": (
                "discrete; report peak_max_position as integer "
                "wavenumber (or discrete category) rather than continuous "
                "offset"
            ),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(
        "All 11 peaks: discrete metric with 10 possible values per peak.\n"
        "Two structural subtypes (single-near-zero, doublet-pair) describe\n"
        "the per-peak grid pattern but share the same manuscript implication.\n"
        "Recommended interpretive convention (any peak):\n"
        "  position_offset < 0.1     peak_max at closest integer\n"
        "  0.1 - 0.3                 one wavenumber from closest integer\n"
        "  0.3 - 0.5                 two wavenumbers from closest integer\n"
        "  0.5 - 0.7                 three wavenumbers from closest integer\n"
        "  0.7 - 0.8                 four wavenumbers from closest integer\n"
        "  > 0.8                     at the tolerance window edge\n"
        "                            (5 wavenumbers; quality flag fires)"
    )
    print()


def print_table_4(
    raw_scores: pd.DataFrame,
    grid_meta: dict[tuple[str, float], dict],
) -> None:
    """E.1 / E.2 documented cases revisited with corrected actual values
    from raw_scores.parquet. The user's E.4 prompt cited 0.81 for sol 996
    Fe2Cl7 326.10; the actual value is 0.3805 (k=328, two integer
    wavenumbers from catalog 326.0975). Pinning the truth here per
    append-only discipline.
    """
    print("=" * 78)
    print(
        "Table 4 - E.1 and E.2 documented cases revisited\n"
        "          (actual position_offset_normalized values from raw_scores;\n"
        "          E.4 prompt cited 0.81 for sol 996 Fe2Cl7 326.10 -- actual 0.3805)"
    )
    print("=" * 78)
    rows = []
    for sol, sclk, seq, pt, compound, ctr, label in SPECIFIC_CASES:
        sub = raw_scores[
            (raw_scores["sol"] == sol)
            & (raw_scores["sclk_int"] == sclk)
            & (raw_scores["seqid"] == seq)
            & (raw_scores["point_number"] == pt)
            & (raw_scores["compound"] == compound)
            & (np.abs(raw_scores["peak_center_catalog"] - ctr) < 0.05)
        ]
        if len(sub) != 1:
            continue
        offset = float(sub["position_offset_normalized"].iloc[0])
        peak_max = float(sub["peak_max_position"].iloc[0])
        m = grid_meta[(compound, ctr)]
        center = m["exact_center"]
        signed_offset_cm = peak_max - center
        if offset < 0.1:
            bucket = "closest integer"
        elif offset < 0.3:
            bucket = "1 wavenumber away"
        elif offset < 0.5:
            bucket = "2 wavenumbers away"
        elif offset < 0.7:
            bucket = "3 wavenumbers away"
        elif offset < 0.8:
            bucket = "4 wavenumbers away"
        else:
            bucket = "tolerance edge (5 wavenumbers; flag fires)"
        rows.append({
            "case": label,
            "peak_max_position": peak_max,
            "exact_center": center,
            "signed_offset_cm": signed_offset_cm,
            "position_offset_normalized": offset,
            "interpretive_bucket": bucket,
            "pattern": m["pattern"],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()


def report_reading_prompt(
    arrays: dict[tuple[str, float], np.ndarray],
    grid_meta: dict[tuple[str, float], dict],
) -> None:
    """Compact summary for the chat-side reading: discrete-not-coarse
    reframing, the >50% quality-flag firing rate per peak, and a
    per-peak n_grid_values + pattern table.
    """
    print("=" * 78)
    print("Reading prompt - position_offset_normalized as discrete metric")
    print("=" * 78)
    print(
        "Discrete-not-coarse reframing: every A-peak has exactly 10 possible\n"
        "values for position_offset_normalized. Each value corresponds to a\n"
        "specific integer peak_max_position. Two grid patterns "
        f"(single-near-zero, doublet-pair) classified at the {DOUBLET_PAIR_GAP_CUTOFF}\n"
        "cutoff between the two smallest grid values. Both patterns share the\n"
        "same manuscript implication: report peak_max_position directly."
    )
    print()
    n_doublet = sum(
        1 for (c, p, _sub) in A_PEAKS
        if grid_meta[(c, p)]["pattern"] == "doublet-pair"
    )
    n_single = sum(
        1 for (c, p, _sub) in A_PEAKS
        if grid_meta[(c, p)]["pattern"] == "single-near-zero"
    )
    print(
        f"Per-peak grid pattern across 11 A-peaks: "
        f"{n_doublet} doublet-pair, {n_single} single-near-zero."
    )
    print()
    print(
        "Per-peak quality-flag firing rate (position_offset_normalized > "
        f"{POSITION_OFFSET_QUALITY_FLAG_THRESHOLD}):"
    )
    for compound, ctr, _sub in A_PEAKS:
        arr = arrays[(compound, ctr)]
        n = len(arr)
        n_above = int((arr > POSITION_OFFSET_QUALITY_FLAG_THRESHOLD).sum())
        pct = n_above / n * 100.0
        print(
            f"  {short_label(compound):<7} {ctr:>7.2f}  "
            f"{pct:6.2f}%  ({n_above}/{n})"
        )
    print()


def plot_panel(
    ax,
    compound: str,
    ctr: float,
    sub: str,
    arr: np.ndarray,
    grid: list[float],
    pattern: str,
    min_non_zero: float,
) -> None:
    color = COMPOUND_COLOR[compound]

    # Light histogram overlay (alpha 0.18, 50 bins) for visual continuity.
    # The discrete grid carries the structural information; the histogram
    # is decorative reinforcement.
    ax.hist(
        arr, bins=np.linspace(0.0, 1.05, 51),
        color=color, alpha=0.18, edgecolor="none",
    )

    # Lollipops at each grid value, height = empirical count.
    counts = empirical_count_at_grid(arr, grid)
    grid_x = np.array(sorted(counts.keys()))
    grid_y = np.array([counts[g] for g in grid_x])
    ax.vlines(grid_x, 0, grid_y, color=color, lw=1.5, alpha=0.95)
    ax.scatter(
        grid_x, grid_y, color=color, s=24, zorder=5,
        edgecolor="black", linewidth=0.4,
    )

    # 0.8 quality-flag threshold and min-non-zero-offset reference verticals.
    ax.axvline(POSITION_OFFSET_QUALITY_FLAG_THRESHOLD, **QFLAG_STYLE)
    ax.axvline(min_non_zero, **MIN_OFFSET_STYLE)

    n = len(arr)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    pct_above = (
        float((arr > POSITION_OFFSET_QUALITY_FLAG_THRESHOLD).sum())
        / n * 100.0
    )

    ax.set_xlim(-0.02, 1.05)
    ax.set_xlabel("position_offset_normalized", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(
        f"{short_label(compound)} {ctr:.2f} ({sub})  -  {pattern}",
        fontsize=9,
    )
    annot = (
        f"n={n}  min_off={min_non_zero:.4f}\n"
        f"p50={p50:.3f}  p95={p95:.3f}\n"
        f"pct>0.8 = {pct_above:.1f}%"
    )
    ax.text(
        0.97, 0.97, annot,
        transform=ax.transAxes, fontsize=7,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
    )


def make_plot(
    arrays: dict[tuple[str, float], np.ndarray],
    grid_meta: dict[tuple[str, float], dict],
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

    for idx, (compound, ctr, sub) in enumerate(A_PEAKS):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col])
        m = grid_meta[(compound, ctr)]
        plot_panel(
            ax, compound, ctr, sub,
            arrays[(compound, ctr)],
            m["grid"], m["pattern"], m["min_non_zero"],
        )

    ax_legend = fig.add_subplot(gs[2, 3])
    ax_legend.axis("off")
    handles = [
        plt.Line2D([0], [0], color="gray", lw=1.5, alpha=0.95,
                   marker="o", markersize=6, markerfacecolor="gray",
                   markeredgecolor="black", markeredgewidth=0.4,
                   label="grid lollipop (empirical count at grid value)"),
        plt.Line2D([0], [0], color="gray", lw=4, alpha=0.18,
                   label="histogram (50 bins; decorative)"),
        plt.Line2D([0], [0],
                   color=QFLAG_STYLE["color"],
                   lw=QFLAG_STYLE["lw"],
                   ls=QFLAG_STYLE["ls"],
                   label=(
                       f"quality-flag threshold "
                       f"({POSITION_OFFSET_QUALITY_FLAG_THRESHOLD})"
                   )),
        plt.Line2D([0], [0],
                   color=MIN_OFFSET_STYLE["color"],
                   lw=MIN_OFFSET_STYLE["lw"],
                   ls=MIN_OFFSET_STYLE["ls"],
                   label="min non-zero offset (closest integer)"),
    ]
    ax_legend.legend(
        handles=handles, loc="center", fontsize=8,
        frameon=True, framealpha=0.9,
    )
    ax_legend.set_title(
        "Plot elements", fontsize=9, pad=10,
    )

    fig.suptitle(
        "Commit E.4: position_offset_normalized per-peak quantization grids",
        fontsize=11, y=0.965,
    )
    return fig


def main() -> int:
    if not RAW_SCORES_PATH.is_file():
        raise FileNotFoundError(f"Missing input: {RAW_SCORES_PATH}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw_scores = load_raw_scores()

    # Sanity: peak_max_position should be integer-valued (the matcher's
    # argmax is over the integer 1.0 cm-1 grid). If non-integer values
    # appear, the discrete-grid analysis breaks; fail fast.
    peak_max = raw_scores["peak_max_position"].to_numpy()
    n_non_int = int(((peak_max != peak_max.astype(int))).sum())
    if n_non_int > 0:
        raise ValueError(
            f"peak_max_position has {n_non_int} non-integer values; "
            f"discrete-grid analysis assumes integer-valued argmax"
        )

    # Sanity: each A-peak should have exactly 1,436 rows in raw_scores.
    arrays = per_peak_arrays(raw_scores)
    for key, arr in arrays.items():
        if len(arr) != 1436:
            raise ValueError(
                f"peak {key} has {len(arr)} rows in raw_scores; "
                f"expected 1436"
            )

    grid_meta = print_table_1(raw_scores)
    print_table_2(arrays, grid_meta)
    print_table_3(grid_meta, arrays)
    print_table_4(raw_scores, grid_meta)
    report_reading_prompt(arrays, grid_meta)

    fig = make_plot(arrays, grid_meta)
    fig.savefig(FIG_PNG, dpi=300)
    fig.savefig(FIG_PDF)
    plt.close(fig)
    print(f"Wrote {FIG_PNG} ({FIG_PNG.stat().st_size} bytes)")
    print(f"Wrote {FIG_PDF} ({FIG_PDF.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
