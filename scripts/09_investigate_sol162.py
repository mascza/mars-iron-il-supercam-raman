"""Commit E.2 investigation: sol 162 EMIM-FeBr4 489.96 cm-1 SNR 17.1 hit.

Sol 162 (target 'guillaumes_162_scam', target_classification 'abraded')
produces SNR 17.12 at the FeBr4 A-secondary 489.96 cm-1 catalog peak on
point 1 of a 9-point Raman raster (seqid 'scam02162'). This is the
highest single-peak SNR in raw_scores.parquet at 489.96. The same
observation already carries a catalog-level confound annotation
(293.64:hematite_293) but no logged mineral confound at 489. The
investigation tests whether 489.96 SNR 17.12 reflects real FeBr4
secondary signal (which would require corroborating peer-peak signatures,
especially the A-primary 218.40 cm-1 [FeBr4]- nu1 mode) or an isolated
baseline / flat-fit / unlogged-mineral artifact.

Pattern follows scripts/08_investigate_sol996.py per the E.1 hand-off
note. Same investigation primitives, same plot scaffold, same corpus-
position diagnostic shape. Differences from E.1: corpus subset is
target_classification == 'abraded' (sol 162's class) rather than
SCCT_Diamond; bottom row is 1x5 (5 FeBr4 A peaks) rather than 1x6;
top panel adds hematite peak overlays from MARS_MINERAL_CONFOUNDS;
tables include peak_height_detrended alongside snr_detrended to surface
the flat-baseline-denominator phenomenon (sol 162 pt 1 has SNR 17.12
driven by local_mad 0.0016, but actual peak height is smaller than
several other abraded observations with cleaner geometry).

Inputs:
  results/matches.parquet
  results/raw_scores.parquet
  data/mars_processed/mars_processed.parquet
  data/reference_library/reference_spectra.parquet
  data/reference_library/peak_catalog.parquet

Outputs:
  docs/figures/e_investigations/sol162_febr4_489.png  (300 dpi)
  docs/figures/e_investigations/sol162_febr4_489.pdf  (vector)
  Numerical tables to stdout (4 tables plus a reading prompt summary).

Scope: commit E.2 of 5. Investigation only; no matcher, catalog, or
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
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _processing import (  # noqa: E402
    MARS_MINERAL_CONFOUNDS,
    compute_local_polynomial_detrending,
)

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
FIG_PNG = FIG_DIR / "sol162_febr4_489.png"
FIG_PDF = FIG_DIR / "sol162_febr4_489.pdf"

# Compound naming and catalog tolerance setup, mirrored from 06/07/08.
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

POLY_DEGREE = 2
HALF_WINDOW = 30.0
PEER_TOL = 5.0
COSINE_HALF_WINDOW = 15.0

# Target observation: sol 162 point 1 of the 9-point scam02162 raster
# on guillaumes_162_scam (abraded). The observation that produced
# SNR 17.12 at FeBr4 489.96 in raw_scores.parquet.
TARGET_SOL = 162
TARGET_SCLK = 681322174
TARGET_SEQID = "scam02162"
TARGET_POINT = 1
TARGET_NAME = "guillaumes_162_scam"
TARGET_CLASS = "abraded"
TARGET_PEAK = 489.96

# Color palette: Wong color-blind-safe, no red-green pairing. FeSO4
# bluish green for cleaner separation from FeCl4 blue. Same as E.1.
COMPOUND_COLOR = {
    "EMIM-FeCl4": "#0072B2",     # blue
    "EMIM-FeBr4": "#E69F00",     # orange
    "EMIM2-Fe2Cl7": "#CC79A7",   # reddish-purple
    "EMIM2-FeSO4": "#009E73",    # bluish green
}

# FeBr4 catalog Class A peaks, in catalog order. Centers match
# peak_catalog.parquet within 0.05 cm-1 (the tolerance used for
# raw_scores filter joins). 218.40 is the 532 nm A-primary; the v4
# handoff transcription "1 A-primary at ~199" was the 785 nm position
# and is not relevant for SuperCam (532 nm). Verified against
# peak_catalog.parquet during the design-review step.
FEBR4_A_CENTERS = [
    (218.40, "primary",   "FeBr4 218.40 (A-primary, 532 nm nu1)"),
    (293.64, "secondary", "FeBr4 293.64 (hematite-confound)"),
    (391.14, "secondary", "FeBr4 391.14"),
    (441.93, "secondary", "FeBr4 441.93"),
    (489.96, "secondary", "FeBr4 489.96 (headline)"),
]

FIBER_BUMP_LO = 200.0
FIBER_BUMP_HI = 530.0
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
    return (
        (df["sol"].astype("int64") == TARGET_SOL)
        & (df["sclk_int"].astype("int64") == TARGET_SCLK)
        & (df["seqid"] == TARGET_SEQID)
        & (df["point_number"].astype("int64") == TARGET_POINT)
    )


def filter_target_obs(df: pd.DataFrame) -> pd.DataFrame:
    return df[is_target_obs_mask(df)]


def percentile_rank_within(values: np.ndarray, target: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return float("nan")
    return float((arr <= target).sum()) / n * 100.0


def print_table_1(raw_scores: pd.DataFrame) -> None:
    """All sol 162 observations, FeBr4 489.96 SNR + peak_height + flags.
    Surfaces the 9-point raster variation: only point 1 produced SNR 17.12;
    the remaining 8 points span SNR -2.40 to 6.02 with varying flags. If
    489 were real chemistry on this abraded patch all 9 points should be
    consistent.
    """
    print("=" * 78)
    print("Table 1 - Sol 162 raster context: 9 observations, FeBr4 489.96 SNR")
    print("=" * 78)
    sub = raw_scores[
        (raw_scores["sol"] == TARGET_SOL)
        & (raw_scores["compound"] == "EMIM-FeBr4")
        & (np.abs(raw_scores["peak_center_catalog"] - TARGET_PEAK) < 0.05)
    ].copy()
    sub = sub.sort_values(
        ["sol", "sclk_int", "seqid", "point_number"], kind="stable"
    )
    cols = [
        "sol", "sclk_int", "seqid", "point_number",
        "target_name", "target_classification",
        "snr_detrended", "peak_height_detrended",
        "local_mad_detrended", "peak_curvature_sign",
        "position_offset_normalized",
    ]
    print(sub[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    n_obs = len(sub)
    n_above_3 = int((sub["snr_detrended"] >= 3.0).sum())
    n_above_5 = int((sub["snr_detrended"] >= 5.0).sum())
    print()
    print(
        f"sol {TARGET_SOL} raster:  {n_obs} observations on the same target;  "
        f"{n_above_3} above 3sigma at FeBr4 489.96;  "
        f"{n_above_5} above 5sigma."
    )
    print(
        f"Target observation is point {TARGET_POINT} of the raster."
    )
    print()


def print_table_2(raw_scores: pd.DataFrame) -> None:
    """Per-peak FeBr4 A scores at target obs (peer-peak diagnostic).
    peak_height_detrended is read directly from raw_scores; the matcher
    populates this column at write time, so no inline derivation from
    snr_detrended * local_mad_detrended is needed.
    """
    print("=" * 78)
    print(
        "Table 2 - Per-peak raw_scores for target observation\n"
        "          across all 5 EMIM-FeBr4 Class A peaks (peer-peak diagnostic)"
    )
    print("=" * 78)
    sub = filter_target_obs(raw_scores)
    sub = sub[
        (sub["compound"] == "EMIM-FeBr4")
        & (sub["peak_class"] == "A")
    ].sort_values("peak_center_catalog", kind="stable")
    cols = [
        "peak_class", "peak_subclass",
        "peak_center_catalog", "peak_max_position",
        "snr_detrended", "peak_height_detrended",
        "local_mad_detrended",
        "peak_curvature_sign", "position_offset_normalized",
    ]
    print(sub[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()


def print_table_3(raw_scores: pd.DataFrame) -> None:
    """Corpus-position diagnostic for FeBr4 489.96.

    Two subsets:
      - Full corpus (n=1436): is sol 162 the corpus outlier or sitting in
        a structurally elevated tail?
      - Abraded subset (target_classification == 'abraded', n~169): how
        does sol 162 rank within its own target class?

    Option C ('hematite-flagged observations') was dropped during design
    review: confound annotation in matches.parquet is catalog-level
    (every FeBr4 row carries the same 293.64:hematite_293 label), so
    there is no observation-level hematite filter to subset on.
    """
    print("=" * 78)
    print(
        "Table 3 - Corpus-position diagnostic for FeBr4 489.96 SNR\n"
        "          (full corpus + abraded subset; Option C dropped per design)"
    )
    print("=" * 78)
    sub = raw_scores[
        (raw_scores["compound"] == "EMIM-FeBr4")
        & (np.abs(raw_scores["peak_center_catalog"] - TARGET_PEAK) < 0.05)
    ].copy()
    target_mask = is_target_obs_mask(sub)
    target_row = sub[target_mask].iloc[0]
    target_snr = float(target_row["snr_detrended"])
    target_height = float(target_row["peak_height_detrended"])
    target_mad = float(target_row["local_mad_detrended"])

    print(
        f"Target  SNR={target_snr:.4f}  "
        f"peak_height_detrended={target_height:.4f}  "
        f"local_mad_detrended={target_mad:.4f}"
    )
    print()

    arr_all = sub["snr_detrended"].to_numpy()
    arr_other = sub.loc[~target_mask, "snr_detrended"].to_numpy()
    nb_full = float(np.max(arr_other)) if len(arr_other) else float("nan")
    print(f"Full corpus (n={len(sub)}):")
    print(
        f"  min={np.min(arr_all):.3f}  p25={np.percentile(arr_all,25):.3f}  "
        f"p50={np.percentile(arr_all,50):.3f}  p75={np.percentile(arr_all,75):.3f}  "
        f"p90={np.percentile(arr_all,90):.3f}  p95={np.percentile(arr_all,95):.3f}  "
        f"p99={np.percentile(arr_all,99):.3f}  max={np.max(arr_all):.3f}"
    )
    print(
        f"  target_pct_in_full_corpus={percentile_rank_within(arr_all, target_snr):.2f}%  "
        f"next_best_other_corpus={nb_full:.4f}  "
        f"gap={target_snr - nb_full:+.4f}"
    )
    print()

    abraded = sub[sub["target_classification"] == TARGET_CLASS].copy()
    n_abr = len(abraded)
    arr_abr = abraded["snr_detrended"].to_numpy()
    abr_other = abraded.loc[
        ~is_target_obs_mask(abraded), "snr_detrended"
    ].to_numpy()
    nb_abr = float(np.max(abr_other)) if len(abr_other) else float("nan")
    print(f"Abraded subset (target_classification == '{TARGET_CLASS}'; n={n_abr}):")
    print(
        f"  min={np.min(arr_abr):.3f}  p25={np.percentile(arr_abr,25):.3f}  "
        f"p50={np.percentile(arr_abr,50):.3f}  p75={np.percentile(arr_abr,75):.3f}  "
        f"p90={np.percentile(arr_abr,90):.3f}  p95={np.percentile(arr_abr,95):.3f}  "
        f"p99={np.percentile(arr_abr,99):.3f}  max={np.max(arr_abr):.3f}"
    )
    print(
        f"  target_pct_in_abraded={percentile_rank_within(arr_abr, target_snr):.2f}%  "
        f"next_best_other_abraded={nb_abr:.4f}  "
        f"gap={target_snr - nb_abr:+.4f}"
    )
    print()

    print("Top 10 abraded SNRs at FeBr4 489.96 (sorted by SNR descending):")
    top_n = abraded.nlargest(10, "snr_detrended").copy()
    cols = [
        "sol", "sclk_int", "seqid", "point_number", "target_name",
        "snr_detrended", "peak_height_detrended",
        "local_mad_detrended", "peak_curvature_sign",
        "position_offset_normalized",
    ]
    print(top_n[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()


def print_table_4(matches: pd.DataFrame) -> None:
    """FeBr4 cosine_diagnostic corpus position. Analog of E.1 Table 4
    but for FeBr4 cosine and abraded subset rather than chloride cosine
    and SCCT subset.
    """
    print("=" * 78)
    print(
        "Table 4 - FeBr4 cosine_diagnostic corpus position\n"
        "          (full corpus + abraded subset; analog of E.1 Table 4)"
    )
    print("=" * 78)
    febr4 = matches[matches["compound"] == "EMIM-FeBr4"].copy()
    target = febr4[is_target_obs_mask(febr4)]
    target_val = float(target["cosine_diagnostic"].iloc[0])

    print(f"Target cosine_diagnostic={target_val:.4f}")
    print()

    full_arr = febr4["cosine_diagnostic"].to_numpy()
    full_other = febr4.loc[
        ~is_target_obs_mask(febr4), "cosine_diagnostic"
    ].to_numpy()
    nb_full = float(np.max(full_other)) if len(full_other) else float("nan")
    print(f"Full corpus (n={len(febr4)}):")
    print(
        f"  min={np.min(full_arr):.4f}  p25={np.percentile(full_arr,25):.4f}  "
        f"p50={np.percentile(full_arr,50):.4f}  p75={np.percentile(full_arr,75):.4f}  "
        f"p90={np.percentile(full_arr,90):.4f}  p95={np.percentile(full_arr,95):.4f}  "
        f"p99={np.percentile(full_arr,99):.4f}  max={np.max(full_arr):.4f}"
    )
    print(
        f"  target_pct={percentile_rank_within(full_arr, target_val):.2f}%  "
        f"next_best_other_corpus={nb_full:.4f}  "
        f"gap={target_val - nb_full:+.4f}"
    )
    print()

    abraded = febr4[febr4["target_classification"] == TARGET_CLASS].copy()
    n_abr = len(abraded)
    abr_arr = abraded["cosine_diagnostic"].to_numpy()
    abr_other = abraded.loc[
        ~is_target_obs_mask(abraded), "cosine_diagnostic"
    ].to_numpy()
    nb_abr = float(np.max(abr_other)) if len(abr_other) else float("nan")
    print(f"Abraded subset (n={n_abr}):")
    print(
        f"  min={np.min(abr_arr):.4f}  p25={np.percentile(abr_arr,25):.4f}  "
        f"p50={np.percentile(abr_arr,50):.4f}  p75={np.percentile(abr_arr,75):.4f}  "
        f"p90={np.percentile(abr_arr,90):.4f}  p95={np.percentile(abr_arr,95):.4f}  "
        f"p99={np.percentile(abr_arr,99):.4f}  max={np.max(abr_arr):.4f}"
    )
    print(
        f"  target_pct={percentile_rank_within(abr_arr, target_val):.2f}%  "
        f"next_best_other_abraded={nb_abr:.4f}  "
        f"gap={target_val - nb_abr:+.4f}"
    )
    print()


def report_reading_prompt(raw_scores: pd.DataFrame) -> None:
    """Headline summary: target SNR + peak_height vs next-best abraded,
    surfacing the SNR-vs-peak-height ranking inversion driven by the
    flat-baseline denominator. Plus peer-peak summary across all 5 FeBr4
    A peaks at sol 162 pt 1, with quality-flag annotation per peak.
    """
    print("=" * 78)
    print(
        "Reading prompt - sol 162 pt 1 vs abraded distribution and\n"
        "                 peer-peak diagnostic across FeBr4 A peaks"
    )
    print("=" * 78)
    sub = raw_scores[
        (raw_scores["compound"] == "EMIM-FeBr4")
        & (np.abs(raw_scores["peak_center_catalog"] - TARGET_PEAK) < 0.05)
    ].copy()
    target_row = sub[is_target_obs_mask(sub)].iloc[0]
    target_snr = float(target_row["snr_detrended"])
    target_height = float(target_row["peak_height_detrended"])
    target_mad = float(target_row["local_mad_detrended"])
    abraded = sub[sub["target_classification"] == TARGET_CLASS].copy()
    abr_other = abraded[~is_target_obs_mask(abraded)]
    nb_other = abr_other.loc[abr_other["snr_detrended"].idxmax()]
    nb_snr = float(nb_other["snr_detrended"])
    nb_height = float(nb_other["peak_height_detrended"])
    nb_mad = float(nb_other["local_mad_detrended"])

    print(
        f"  Target (sol {TARGET_SOL} pt {TARGET_POINT}, {TARGET_NAME}):  "
        f"SNR={target_snr:.2f}  peak_height={target_height:.4f}  "
        f"local_mad={target_mad:.4f}"
    )
    print(
        f"  Next-best abraded (sol {int(nb_other['sol'])} pt "
        f"{int(nb_other['point_number'])}, {nb_other['target_name']}):  "
        f"SNR={nb_snr:.2f}  peak_height={nb_height:.4f}  "
        f"local_mad={nb_mad:.4f}"
    )
    if target_height > 0:
        height_ratio = nb_height / target_height
        print(
            f"  -> Target SNR > Next-best SNR by {target_snr - nb_snr:+.2f}, "
            f"but Target peak_height < Next-best peak_height by ratio "
            f"{height_ratio:.2f}x. SNR ranking inverts peak-height ranking "
            f"due to flat-baseline denominator (target local_mad "
            f"{target_mad:.4f} vs next-best {nb_mad:.4f})."
        )
    print()
    print("  Peer-peak summary (sol 162 pt 1, all 5 FeBr4 A peaks):")
    peer = filter_target_obs(raw_scores)
    peer = peer[
        (peer["compound"] == "EMIM-FeBr4") & (peer["peak_class"] == "A")
    ].sort_values("peak_center_catalog", kind="stable")
    for _, p in peer.iterrows():
        flag_parts: list[str] = []
        if int(p["peak_curvature_sign"]) != -1:
            flag_parts.append("curvature_flag")
        if float(p["position_offset_normalized"]) > 0.8:
            flag_parts.append("offset_flag")
        if float(p["snr_detrended"]) < 3.0:
            flag_parts.append("below_3sigma")
        flag_string = " ".join(flag_parts) if flag_parts else "clean"
        print(
            f"    {p['peak_subclass']:<10}  "
            f"{float(p['peak_center_catalog']):7.2f}  "
            f"SNR={float(p['snr_detrended']):+6.2f}  "
            f"peak_h={float(p['peak_height_detrended']):.4f}  "
            f"curv={int(p['peak_curvature_sign']):+d}  "
            f"offset={float(p['position_offset_normalized']):.3f}  "
            f"[{flag_string}]"
        )
    print()


def plot_window_panel(
    ax,
    mars_wn: np.ndarray,
    mars_y: np.ndarray,
    ctr: float,
    title_prefix: str,
    ref_means: dict[str, tuple[np.ndarray, np.ndarray]],
    catalog: pd.DataFrame,
    raw_scores: pd.DataFrame,
) -> None:
    compound = "EMIM-FeBr4"
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
        label="ref FeBr4 detrended",
    )

    rs = raw_scores[
        is_target_obs_mask(raw_scores)
        & (raw_scores["compound"] == compound)
        & (np.abs(raw_scores["peak_center_catalog"] - ctr) < 0.05)
    ]
    if not rs.empty:
        local_mad = float(rs["local_mad_detrended"].iloc[0])
        snr = float(rs["snr_detrended"].iloc[0])
        curv = int(rs["peak_curvature_sign"].iloc[0])
        offset = float(rs["position_offset_normalized"].iloc[0])
        peak_h = float(rs["peak_height_detrended"].iloc[0])
        five_sig = 5.0 * local_mad
        ax.axhspan(
            -five_sig, five_sig, color="black", alpha=0.06,
            label=f"+/-5 sigma  (local MAD={local_mad:.3f})",
        )
        ax.axhline(five_sig, color="black", lw=0.4, ls=":", alpha=0.6)
        ax.axhline(-five_sig, color="black", lw=0.4, ls=":", alpha=0.6)
        annot = (
            f"SNR={snr:+.2f}  peak_h={peak_h:.4f}  "
            f"curv={curv:+d}  offset={offset:.2f}"
        )
    else:
        annot = "no raw_scores row"

    ax.axvline(ctr, color="gray", ls="--", lw=0.6)
    ax.axvspan(ctr - tol, ctr + tol, color="gray", alpha=0.10, lw=0)
    ax.set_xlim(ctr - COSINE_HALF_WINDOW, ctr + COSINE_HALF_WINDOW)
    ax.set_xlabel("cm$^{-1}$", fontsize=9)
    ax.set_ylabel("detrended", fontsize=9)
    ax.set_title(f"{title_prefix}\n{annot}", fontsize=8)
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
        nrows=2, ncols=5,
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

    for ctr, _sub, _label in FEBR4_A_CENTERS:
        ax_top.axvline(
            ctr, color=COMPOUND_COLOR["EMIM-FeBr4"],
            ls=":", lw=0.7, alpha=0.7,
        )
    ax_top.plot(
        [], [], color=COMPOUND_COLOR["EMIM-FeBr4"],
        ls=":", lw=0.7, alpha=0.7,
        label=(
            "FeBr4 A-peaks: "
            + ", ".join(f"{c:.2f}" for c, _, _ in FEBR4_A_CENTERS)
        ),
    )

    hematite_peaks = MARS_MINERAL_CONFOUNDS["hematite"]
    for hp in hematite_peaks:
        ax_top.axvline(
            float(hp), color="gray", ls="--", lw=0.7, alpha=0.6,
        )
    ax_top.plot(
        [], [], color="gray", ls="--", lw=0.7, alpha=0.6,
        label=(
            "hematite peaks (MARS_MINERAL_CONFOUNDS): "
            + ", ".join(str(p) for p in hematite_peaks)
        ),
    )

    ax_top.set_xlim(150, 1700)
    ax_top.set_xlabel("Raman shift (cm$^{-1}$)")
    ax_top.set_ylabel("intensity_normalized_mean")
    ax_top.set_title(
        (
            f"sol {TARGET_SOL}  {TARGET_SEQID}  point {TARGET_POINT}  -  "
            f"{TARGET_CLASS} ({TARGET_NAME})  -  full spectrum + lab "
            f"overlays + 5 FeBr4 A-peaks (orange dotted) + 7 hematite "
            f"peaks (gray dashed)"
        ),
        fontsize=10,
    )
    ax_top.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)

    for col_idx, (ctr, _sub, label) in enumerate(FEBR4_A_CENTERS):
        ax = fig.add_subplot(gs[1, col_idx])
        plot_window_panel(
            ax, mars_wn, mars_y, ctr, label,
            ref_means, catalog, raw_scores,
        )

    fig.suptitle(
        "Commit E.2: sol 162 EMIM-FeBr4 489.96 SNR 17.1 diagnosis",
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

    print_table_1(raw_scores)
    print_table_2(raw_scores)
    print_table_3(raw_scores)
    print_table_4(matches)
    report_reading_prompt(raw_scores)

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
