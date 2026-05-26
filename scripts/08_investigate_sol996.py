"""Commit E.1 investigation: sol 996 SCCT_Diamond strong_candidate diagnosis.

Two of the 41 strong_candidate rows in matches.parquet sit on a single
sol 996 SCCT_Diamond observation
(sol=996, sclk_int=755369654, seqid='scam05996', point_number=32):
  EMIM-FeCl4    cosine 0.906, full-corpus percentile 98.8
  EMIM2-Fe2Cl7  cosine 0.792, full-corpus percentile 95.8
The same observation produces no_evidence on EMIM-FeBr4 (cosine -0.10) and
EMIM2-FeSO4 (cosine -0.22), so the elevation is specifically on chloride
templates. Diamond cannot contain ionic liquid; the IL diagnostic windows
are far from the 1332 cm-1 diamond peak. The cosine score is being driven
by residual structure in the chloride A-windows after polynomial-2
detrending.

This script characterizes the spectrum and tests whether the result is
(a) a single anomalous observation, (b) a systematic SCCT_Diamond pattern
on chloride templates with sol 996 just clearing the gates, or (c)
something else. The decision is reported in chat after the run; the
script itself prints numerical evidence and saves a diagnostic figure.

Inputs:
  results/matches.parquet
  results/raw_scores.parquet
  data/mars_processed/mars_processed.parquet
  data/reference_library/reference_spectra.parquet
  data/reference_library/peak_catalog.parquet

Outputs:
  docs/figures/e_investigations/sol996_scct.png  (300 dpi)
  docs/figures/e_investigations/sol996_scct.pdf  (vector)
  Numerical tables to stdout (4 tables plus a reading prompt summary).

Scope: commit E.1 of 5. Investigation only; no matcher, catalog, or
parquet changes. The figure directory is tracked (committed artifact).
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
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _processing import compute_local_polynomial_detrending  # noqa: E402

# Inputs.
MATCHES_PATH = REPO_ROOT / "results" / "matches.parquet"
RAW_SCORES_PATH = REPO_ROOT / "results" / "raw_scores.parquet"
MARS_PATH = REPO_ROOT / "data" / "mars_processed" / "mars_processed.parquet"
REF_SPECTRA_PATH = (
    REPO_ROOT / "data" / "reference_library" / "reference_spectra.parquet"
)
CATALOG_PATH = (
    REPO_ROOT / "data" / "reference_library" / "peak_catalog.parquet"
)

# Outputs (tracked).
FIG_DIR = REPO_ROOT / "docs" / "figures" / "e_investigations"
FIG_PNG = FIG_DIR / "sol996_scct.png"
FIG_PDF = FIG_DIR / "sol996_scct.pdf"

# Compound naming and catalog tolerance setup, mirrored from 06/07.
COMPOUND_ORDER = (
    "EMIM-FeCl4",
    "EMIM-FeBr4",
    "EMIM2-Fe2Cl7",
    "EMIM2-FeSO4",
)
TOLERANCE_OVERRIDES: dict[tuple[str, float], float] = {
    ("EMIM2-FeSO4", 1022.01): 2.5,
    ("EMIM2-FeSO4", 1027.38): 2.5,
    ("EMIM2-FeSO4", 1086.60): 2.5,
    ("EMIM2-FeSO4", 1091.81): 2.5,
}
DEFAULT_TOLERANCE = 5.0

# Match the design constants from 06.
POLY_DEGREE = 2
HALF_WINDOW = 30.0
PEER_TOL = 5.0
COSINE_HALF_WINDOW = 15.0

# Target observation: the single sol 996 SCCT_Diamond observation that
# produced both strong_candidate rows in matches.parquet. Verified by
# joining matches.parquet (sol=996, target_name='scct_diamond',
# tier='strong_candidate') against mars_processed.parquet.
TARGET_SOL = 996
TARGET_SCLK = 755369654
TARGET_SEQID = "scam05996"
TARGET_POINT = 32

# Color palette: Wong color-blind-safe, no red-green pairing. FeSO4 swapped
# from sky blue (#56B4E9) to bluish green (#009E73) for clearer separation
# from FeCl4's blue.
COMPOUND_COLOR = {
    "EMIM-FeCl4": "#0072B2",     # blue
    "EMIM-FeBr4": "#E69F00",     # orange
    "EMIM2-Fe2Cl7": "#CC79A7",   # reddish-purple
    "EMIM2-FeSO4": "#009E73",    # bluish green
}

# Bottom-row windows. Panels 1-5 drive the chloride-compound cosine scores
# at sol 996 (FeCl4 cosine_pct 98.8, Fe2Cl7 cosine_pct 95.8). Panel 6 is a
# negative control: FeBr4 A-secondary 489.96 where the cosine is -0.10 and
# Mars should NOT trace the lab reference. The 489.96 panel is labeled
# explicitly so it is not confused with the sol 162 EMIM-FeBr4 489.96
# investigation in commit E.2 (different observation, different question).
PANEL_WINDOWS = [
    ("EMIM2-Fe2Cl7", 190.94, "Fe2Cl7 190.94 (Fe-Cl-Fe bridge)"),
    ("EMIM2-Fe2Cl7", 326.10, "Fe2Cl7 326.10"),
    ("EMIM-FeCl4",   330.52, "FeCl4 330.52"),
    ("EMIM-FeCl4",   386.31, "FeCl4 386.31"),
    ("EMIM2-Fe2Cl7", 460.31, "Fe2Cl7 460.31"),
    ("EMIM-FeBr4",   489.96, "FeBr4 489.96 (negative control)"),
]

FIBER_BUMP_LO = 200.0
FIBER_BUMP_HI = 530.0

# Lab-reference scale factor for the top panel overlay. Lab spectra are
# already max-normalized per compound; multiplying by 0.6 plots them at
# ~60 percent of the Mars peak height so the diamond peak still dominates
# the eye and the catalog overlays stay legible.
LAB_REF_TOP_PANEL_SCALE = 0.6


def tolerance_for(compound: str, center: float) -> float:
    for (cmp, ctr), v in TOLERANCE_OVERRIDES.items():
        if cmp == compound and abs(ctr - center) < 0.05:
            return v
    return DEFAULT_TOLERANCE


def load_catalog() -> pd.DataFrame:
    df = pd.read_parquet(CATALOG_PATH)
    df = df[
        (df["lab_or_convolved"] == "convolved") & (df["is_aggregate"] == True)
    ].copy()
    df["tolerance_used"] = [
        tolerance_for(c, p)
        for c, p in zip(df["compound"], df["position_cm_inv"])
    ]

    def parse_subclass(label: object, notes: object) -> str:
        if label != "A":
            return ""
        s = "" if notes is None else str(notes)
        if s.startswith("primary;"):
            return "primary"
        if s.startswith("secondary;"):
            return "secondary"
        return ""

    df["peak_subclass"] = [
        parse_subclass(lbl, n)
        for lbl, n in zip(df["class_label"], df["class_notes"])
    ]
    return df.sort_values(
        ["compound", "position_cm_inv"], kind="stable"
    ).reset_index(drop=True)


def load_reference_means(
    compound_order: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    df = pd.read_parquet(REF_SPECTRA_PATH)
    df = df[df["usable_for_supercam"] == True].copy()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in compound_order:
        sub = df[df["compound"] == c]
        agg = (
            sub.groupby("wavenumber_cm_inv")["intensity_normalized_convolved"]
            .mean()
            .reset_index()
            .sort_values("wavenumber_cm_inv", kind="stable")
        )
        out[c] = (
            agg["wavenumber_cm_inv"].to_numpy(dtype=float),
            agg["intensity_normalized_convolved"].to_numpy(dtype=float),
        )
    return out


def build_peer_centers_for_window(
    catalog: pd.DataFrame, target_center: float,
    half_window: float = HALF_WINDOW,
) -> list[float]:
    W_lo = target_center - half_window
    W_hi = target_center + half_window
    out: list[float] = []
    for c, tol in zip(catalog["position_cm_inv"], catalog["tolerance_used"]):
        c = float(c)
        tol = float(tol)
        if c + tol < W_lo or c - tol > W_hi:
            continue
        if abs(c - target_center) < 1e-3:
            continue
        out.append(c)
    return out


def load_target_mars_spectrum(
    mars: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    sub = mars[
        (mars["sol"] == TARGET_SOL)
        & (mars["sclk_int"] == TARGET_SCLK)
        & (mars["seqid"] == TARGET_SEQID)
        & (mars["point"] == TARGET_POINT)
    ].sort_values("wavenumber_cm_inv", kind="stable")
    if sub.empty:
        raise RuntimeError(
            f"target observation not found in mars_processed: "
            f"({TARGET_SOL}, {TARGET_SCLK}, {TARGET_SEQID}, {TARGET_POINT})"
        )
    wn = sub["wavenumber_cm_inv"].to_numpy(dtype=float)
    y = sub["intensity_normalized_mean"].to_numpy(dtype=float)
    return wn, y


def is_target_obs_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting the target observation rows in any of the
    output parquets (matches, raw_scores). All four key columns are int64
    or string already in those files.
    """
    return (
        (df["sol"].astype("int64") == TARGET_SOL)
        & (df["sclk_int"].astype("int64") == TARGET_SCLK)
        & (df["seqid"] == TARGET_SEQID)
        & (df["point_number"].astype("int64") == TARGET_POINT)
    )


def filter_target_obs(df: pd.DataFrame) -> pd.DataFrame:
    return df[is_target_obs_mask(df)]


def percentile_rank_within(
    values: np.ndarray, target: float
) -> float:
    n = len(values)
    if n == 0:
        return float("nan")
    return float((values <= target).sum()) / n * 100.0


def print_table_1(matches: pd.DataFrame) -> None:
    print("=" * 78)
    print("Table 1 - SCCT_Diamond rows at sol 996")
    print("=" * 78)
    sub = matches[
        (matches["sol"] == TARGET_SOL)
        & (matches["target_name"] == "scct_diamond")
    ].copy()
    distinct_obs = sub[
        ["sol", "sclk_int", "seqid", "point_number"]
    ].drop_duplicates(
        subset=["sol", "sclk_int", "seqid", "point_number"]
    )
    print(f"matches.parquet rows: {len(sub)}")
    print(f"distinct observations: {len(distinct_obs)}")
    print()
    cols = [
        "sol", "sclk_int", "seqid", "point_number",
        "target_name", "target_classification",
        "compound", "tier",
        "cosine_diagnostic",
        "cosine_percentile_within_compound_full_corpus",
    ]
    sorted_sub = sub.sort_values(
        ["sol", "sclk_int", "seqid", "point_number", "compound"],
        kind="stable",
    )
    print(
        sorted_sub[cols].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print()


def print_table_2(raw_scores: pd.DataFrame) -> None:
    print("=" * 78)
    print(
        "Table 2 - Per-peak raw_scores for target observation\n"
        "          (EMIM-FeCl4 and EMIM2-Fe2Cl7, Class A peaks only)"
    )
    print("=" * 78)
    sub = filter_target_obs(raw_scores)
    sub = sub[
        sub["compound"].isin(["EMIM-FeCl4", "EMIM2-Fe2Cl7"])
        & (sub["peak_class"] == "A")
    ].sort_values(
        ["compound", "peak_center_catalog"], kind="stable"
    )
    cols = [
        "compound", "peak_class", "peak_subclass",
        "peak_center_catalog", "peak_max_position",
        "snr_detrended", "peak_curvature_sign",
        "position_offset_normalized", "local_mad_detrended",
    ]
    print(
        sub[cols].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print()


def print_table_3(matches: pd.DataFrame) -> None:
    print("=" * 78)
    print("Table 3 - Cosine_diagnostic for target observation, all 4 compounds")
    print("=" * 78)
    sub = filter_target_obs(matches).copy()
    order_idx = {k: i for i, k in enumerate(COMPOUND_ORDER)}
    sub["_order"] = sub["compound"].map(order_idx)
    sub = sub.sort_values("_order", kind="stable").drop(columns="_order")
    cols = [
        "compound",
        "cosine_diagnostic",
        "cosine_percentile_within_compound_full_corpus",
        "cosine_percentile_within_compound_science_only",
        "tier",
    ]
    print(
        sub[cols].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print()


def print_table_4(matches: pd.DataFrame) -> None:
    print("=" * 78)
    print("Table 4 - Cosine distribution within SCCT_Diamond corpus, per compound")
    print("=" * 78)
    scct = matches[matches["target_name"] == "scct_diamond"].copy()
    target_mask_in_scct = is_target_obs_mask(scct)
    target = filter_target_obs(matches)
    rows = []
    for c in COMPOUND_ORDER:
        sub_all = (
            scct[scct["compound"] == c]["cosine_diagnostic"]
            .sort_values(kind="stable")
            .to_numpy()
        )
        # SCCT cosine values excluding the target observation. The next-best
        # cosine is taken from this excluded set, so the gap to the target
        # is meaningful regardless of whether the target is itself the
        # corpus max. Negative gap = target is not the corpus outlier.
        sub_other = (
            scct[
                (scct["compound"] == c) & (~target_mask_in_scct)
            ]["cosine_diagnostic"]
            .to_numpy()
        )
        n = len(sub_all)
        target_val = float(
            target[target["compound"] == c]["cosine_diagnostic"].iloc[0]
        )
        pct = percentile_rank_within(sub_all, target_val)
        next_best_other = (
            float(np.max(sub_other)) if len(sub_other) > 0 else float("nan")
        )
        rows.append(
            {
                "compound": c,
                "n_scct": int(n),
                "min": float(np.min(sub_all)) if n else float("nan"),
                "p25": float(np.percentile(sub_all, 25)) if n else float("nan"),
                "p50": float(np.percentile(sub_all, 50)) if n else float("nan"),
                "p75": float(np.percentile(sub_all, 75)) if n else float("nan"),
                "p90": float(np.percentile(sub_all, 90)) if n else float("nan"),
                "p95": float(np.percentile(sub_all, 95)) if n else float("nan"),
                "p99": float(np.percentile(sub_all, 99)) if n else float("nan"),
                "max": float(np.max(sub_all)) if n else float("nan"),
                "sol996_cosine": target_val,
                "sol996_pct_in_scct": pct,
                "next_best_other_scct": next_best_other,
            }
        )
    df = pd.DataFrame(rows)
    print(
        df.to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print()


def report_reading_prompt(matches: pd.DataFrame) -> None:
    """One line per chloride compound. Reports target cosine, SCCT-only
    percentile, the next-best cosine in SCCT excluding the target, and the
    gap between target and next-best. Negative gap means the target is not
    the corpus outlier on that compound; positive gap means it is, and the
    magnitude indicates how isolated. The reading is reported in chat.
    """
    print("=" * 78)
    print(
        "Reading prompt - sol 996 chloride cosines vs SCCT-only distribution"
    )
    print("=" * 78)
    scct = matches[matches["target_name"] == "scct_diamond"].copy()
    target_mask = is_target_obs_mask(scct)
    target = filter_target_obs(matches)
    for c in ("EMIM-FeCl4", "EMIM2-Fe2Cl7"):
        sub_all = scct[scct["compound"] == c]["cosine_diagnostic"].to_numpy()
        sub_other = (
            scct[(scct["compound"] == c) & (~target_mask)]
            ["cosine_diagnostic"].to_numpy()
        )
        target_val = float(
            target[target["compound"] == c]["cosine_diagnostic"].iloc[0]
        )
        n = len(sub_all)
        if n == 0:
            continue
        pct = percentile_rank_within(sub_all, target_val)
        next_best = (
            float(np.max(sub_other)) if len(sub_other) > 0 else float("nan")
        )
        gap = target_val - next_best
        print(
            f"  {c:<14}  cosine={target_val:+.4f}  "
            f"SCCT pct={pct:5.1f}%  next-best (other SCCT)={next_best:+.4f}  "
            f"gap={gap:+.4f}"
        )
    print()


def plot_window_panel(
    ax,
    mars_wn: np.ndarray,
    mars_y: np.ndarray,
    compound: str,
    ctr: float,
    label: str,
    ref_means: dict[str, tuple[np.ndarray, np.ndarray]],
    catalog: pd.DataFrame,
    raw_scores: pd.DataFrame,
) -> None:
    tol = float(tolerance_for(compound, ctr))
    peers = build_peer_centers_for_window(catalog, ctr)

    mars_result = compute_local_polynomial_detrending(
        mars_wn, mars_y, ctr, tol, peers,
        half_window=COSINE_HALF_WINDOW, peer_tol=PEER_TOL,
        poly_degree=POLY_DEGREE,
    )
    mars_x = mars_result["wavenumbers_window"]
    mars_d = mars_result["detrended_window"]

    ref_wn, ref_y = ref_means[compound]
    ref_result = compute_local_polynomial_detrending(
        ref_wn, ref_y, ctr, tol, peers,
        half_window=COSINE_HALF_WINDOW, peer_tol=PEER_TOL,
        poly_degree=POLY_DEGREE,
    )
    ref_x = ref_result["wavenumbers_window"]
    ref_d = ref_result["detrended_window"]

    ax.plot(
        mars_x, mars_d, color="black", lw=0.9, marker="o", ms=2.5,
        label="Mars detrended",
    )
    ax.plot(
        ref_x, ref_d, color=COMPOUND_COLOR[compound], lw=1.0,
        label="ref detrended",
    )

    rs = raw_scores[
        is_target_obs_mask(raw_scores)
        & (raw_scores["compound"] == compound)
        & (np.abs(raw_scores["peak_center_catalog"] - ctr) < 0.05)
    ]
    if not rs.empty:
        local_mad = float(rs["local_mad_detrended"].iloc[0])
        snr = float(rs["snr_detrended"].iloc[0])
        five_sig = 5.0 * local_mad
        ax.axhspan(
            -five_sig, five_sig, color="black", alpha=0.06,
            label=f"+/-5 sigma  (local MAD={local_mad:.3f})",
        )
        ax.axhline(five_sig, color="black", lw=0.4, ls=":", alpha=0.6)
        ax.axhline(-five_sig, color="black", lw=0.4, ls=":", alpha=0.6)
        snr_str = f"SNR={snr:+.2f}"
    else:
        snr_str = "SNR=n/a"

    ax.axvline(ctr, color="gray", ls="--", lw=0.6)
    ax.axvspan(ctr - tol, ctr + tol, color="gray", alpha=0.10, lw=0)
    ax.set_xlim(ctr - COSINE_HALF_WINDOW, ctr + COSINE_HALF_WINDOW)
    ax.set_xlabel("cm$^{-1}$", fontsize=9)
    ax.set_ylabel("detrended", fontsize=9)
    ax.set_title(f"{label}\n{snr_str}", fontsize=8)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.9)
    ax.tick_params(axis="both", labelsize=8)


def make_plot(
    mars_wn: np.ndarray,
    mars_y: np.ndarray,
    ref_means: dict[str, tuple[np.ndarray, np.ndarray]],
    catalog: pd.DataFrame,
    raw_scores: pd.DataFrame,
):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig = plt.figure(figsize=(14.0, 9.0))
    gs = fig.add_gridspec(
        nrows=2, ncols=6,
        height_ratios=[1.0, 1.4],
        left=0.06, right=0.98, top=0.93, bottom=0.08,
        hspace=0.50, wspace=0.34,
    )
    ax_top = fig.add_subplot(gs[0, :])

    mars_max = float(np.nanmax(mars_y))
    ax_top.axvspan(
        FIBER_BUMP_LO, FIBER_BUMP_HI,
        color="#FFD700", alpha=0.10,
        label=(
            f"fiber-bump {int(FIBER_BUMP_LO)}-{int(FIBER_BUMP_HI)} cm$^{{-1}}$"
        ),
    )
    for c in COMPOUND_ORDER:
        wn_r, y_r = ref_means[c]
        ref_max = float(np.nanmax(y_r))
        scaled = y_r * (LAB_REF_TOP_PANEL_SCALE * mars_max / ref_max)
        ax_top.plot(
            wn_r, scaled, color=COMPOUND_COLOR[c], lw=0.6, alpha=0.45,
            label=(
                f"ref {c}  (scaled to "
                f"{LAB_REF_TOP_PANEL_SCALE:.2f} x Mars max)"
            ),
        )
    ax_top.plot(
        mars_wn, mars_y, color="black", lw=0.9,
        label=(
            f"Mars sol {TARGET_SOL} {TARGET_SEQID} (point {TARGET_POINT})"
        ),
        zorder=10,
    )

    cat_a = catalog[catalog["class_label"] == "A"]
    n_a = len(cat_a)
    for _, p in cat_a.iterrows():
        ctr = float(p["position_cm_inv"])
        col = COMPOUND_COLOR[p["compound"]]
        ax_top.axvline(ctr, color=col, ls=":", lw=0.5, alpha=0.55)

    diamond_in_range = (mars_wn >= 1320) & (mars_wn <= 1345)
    if diamond_in_range.any():
        masked = np.where(diamond_in_range, mars_y, -np.inf)
        idx = int(np.nanargmax(masked))
        d_x = float(mars_wn[idx])
        d_y = float(mars_y[idx])
        ax_top.annotate(
            (
                f"diamond peak  {d_x:.0f} cm$^{{-1}}$  "
                f"intensity_normalized_mean = {d_y:.3f} (unit scale)"
            ),
            xy=(d_x, d_y),
            xytext=(d_x - 320.0, d_y - 0.18),
            fontsize=8, color="black",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5, alpha=0.7),
        )

    ax_top.set_xlim(150, 1700)
    ax_top.set_xlabel("Raman shift (cm$^{-1}$)")
    ax_top.set_ylabel("intensity_normalized_mean")
    ax_top.set_title(
        (
            f"sol {TARGET_SOL}  {TARGET_SEQID}  point {TARGET_POINT}  "
            f"-  SCCT_Diamond  -  full spectrum + lab overlays  "
            f"({n_a} catalog Class A peak positions marked)"
        ),
        fontsize=10,
    )
    ax_top.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)

    for col_idx, (compound, ctr, label) in enumerate(PANEL_WINDOWS):
        ax = fig.add_subplot(gs[1, col_idx])
        plot_window_panel(
            ax, mars_wn, mars_y, compound, ctr, label,
            ref_means, catalog, raw_scores,
        )

    fig.suptitle(
        "Commit E.1: sol 996 SCCT_Diamond strong_candidate diagnosis",
        fontsize=11, y=0.985,
    )
    return fig


def main() -> int:
    for p in (
        MATCHES_PATH, RAW_SCORES_PATH,
        MARS_PATH, REF_SPECTRA_PATH, CATALOG_PATH,
    ):
        if not p.is_file():
            raise FileNotFoundError(f"Missing input: {p}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    matches = pd.read_parquet(MATCHES_PATH)
    raw_scores = pd.read_parquet(RAW_SCORES_PATH)
    mars = pd.read_parquet(MARS_PATH)
    catalog = load_catalog()
    if len(catalog) != 46:
        raise ValueError(
            f"expected 46 aggregate-convolved catalog rows, got {len(catalog)}"
        )
    ref_means = load_reference_means(COMPOUND_ORDER)

    print_table_1(matches)
    print_table_2(raw_scores)
    print_table_3(matches)
    print_table_4(matches)
    report_reading_prompt(matches)

    mars_wn, mars_y = load_target_mars_spectrum(mars)
    fig = make_plot(mars_wn, mars_y, ref_means, catalog, raw_scores)
    fig.savefig(FIG_PNG, dpi=300)
    fig.savefig(FIG_PDF)
    plt.close(fig)
    print(f"Wrote {FIG_PNG} ({FIG_PNG.stat().st_size} bytes)")
    print(f"Wrote {FIG_PDF} ({FIG_PDF.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
