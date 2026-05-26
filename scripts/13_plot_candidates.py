"""Commit F.1: plot every named candidate (passes + partials) for manuscript
review. One PDF per row, two stacked panels:

  Top panel    — raw intensity_normalized_mean over 150-1700 cm-1 (the column
                 the matcher in 06_match_mars.py read), with per-A-peak local
                 poly-2 fit overlays drawn over each peak's +/- HALF_WINDOW
                 fit window, all catalog A-peak positions as vertical lines
                 (gate-driving primary thicker), all class-B (cation) peak
                 positions as thin grey lines, R3 zone (+/- 2.5 cm-1) and
                 cosine half-window (+/- 15 cm-1) shaded around the
                 gate-driving primary catalog center, an annotation box with
                 tier/pass/reason/gdp/cation/cosine/qflags.

  Bottom panel — per-A-peak detrended residual stitched only inside each
                 peak's +/- HALF_WINDOW fit window, with horizontal +3*sigma
                 and +5*sigma lines per peak (sigma = local_mad_detrended
                 from raw_scores.parquet, the same MAD the matcher used in
                 _processing.compute_local_polynomial_detrending). The
                 gate-driving primary's argmax is marked with a triangle
                 and a vertical line; only the GDP segment's sigma lines are
                 labeled "3sigma"/"5sigma" to avoid clutter.

Reads (no writes outside results/figures/candidate_spectra/):
  results/strong_and_candidate_classification.parquet
  results/raw_scores.parquet
  data/reference_library/peak_catalog.parquet
  data/mars_processed/mars_processed.parquet

Writes:
  results/figures/candidate_spectra/headlines/        4 PDFs
  results/figures/candidate_spectra/passes/         329 PDFs
  results/figures/candidate_spectra/partials/       146 PDFs
  results/figures/candidate_spectra/combined/        4 multi-page PDFs
      combined_passes_FeCl4.pdf    (92 pages)
      combined_passes_FeBr4.pdf   (100 pages)
      combined_passes_Fe2Cl7.pdf  (137 pages)
      combined_partials_FeSO4.pdf (146 pages)
  results/figures/candidate_spectra/INDEX.csv

Headlines (curated in headlines/) are also retained in passes/ or partials/
per their pass_label, so passes/ + partials/ collectively cover all 475
named candidates.

Determinism: rows iterated in stable sort order (compound, pass_label, sol,
point_number, gdp_signed_offset_cm). Combined PDFs accumulate pages in the
same order. Single-thread.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parent.parent
SUPP_PATH = REPO / "results" / "strong_and_candidate_classification.parquet"
RAW_PATH = REPO / "results" / "raw_scores.parquet"
CATALOG_PATH = REPO / "data" / "reference_library" / "peak_catalog.parquet"
MARS_PATH = REPO / "data" / "mars_processed" / "mars_processed.parquet"
OUT_ROOT = REPO / "results" / "figures" / "candidate_spectra"

# Constants from _processing.py and session log E.5. The matcher in
# 06_match_mars.py uses HALF_WINDOW = 30 (per-peak fit) and
# COSINE_HALF_WINDOW = 15. R3 zone is the manuscript-framing tightening
# documented in session_log.md E.5 lines 1311-1313.
HALF_WINDOW = 30.0
COSINE_HALF_WINDOW = 15.0
R3_HALF_WINDOW = 2.5

# Wong palette per compound; B (cation) peaks neutral grey to avoid
# red-green pairings inside any single panel.
COMPOUND_COLOR = {
    "EMIM-FeCl4": "#0072B2",
    "EMIM-FeBr4": "#E69F00",
    "EMIM2-Fe2Cl7": "#CC79A7",
    "EMIM2-FeSO4": "#009E73",
}
B_PEAK_COLOR = "#555555"

COMPOUND_SHORT = {
    "EMIM-FeCl4": "FeCl4",
    "EMIM-FeBr4": "FeBr4",
    "EMIM2-Fe2Cl7": "Fe2Cl7",
    "EMIM2-FeSO4": "FeSO4",
}

# Headline rows from session log E.5 "Cleanest named candidate per compound".
# Tuple key: (compound, sol, sclk_int, point_number). sclk_int included to
# disambiguate multi-acquisition observations (e.g. FeSO4 sol 1660 pt 1 has
# both a fails row at sclk 814307430 and the partial headline at sclk
# 814307790). Sclk_int values looked up from
# results/strong_and_candidate_classification.parquet after F.0 regeneration.
HEADLINES: set[tuple[str, int, int, int]] = {
    ("EMIM-FeCl4", 1154, 769385046, 1),
    ("EMIM-FeBr4", 247, 688865041, 1),
    ("EMIM2-Fe2Cl7", 268, 690733117, 1),
    ("EMIM2-FeSO4", 1660, 814307790, 1),
}


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def safe_name(s: str, max_len: int = 32) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return (s[:max_len] or "unknown").rstrip("_")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    supp = pd.read_parquet(SUPP_PATH)
    raw = pd.read_parquet(RAW_PATH)
    cat = pd.read_parquet(CATALOG_PATH)
    cat = cat[(cat["is_aggregate"]) & (cat["class_label"] != "ARTIFACT")].copy()
    mars = pd.read_parquet(
        MARS_PATH,
        columns=[
            "sol", "sclk_int", "seqid", "point",
            "wavenumber_cm_inv", "intensity_normalized_mean",
        ],
    )
    return supp, raw, cat, mars


def index_spectra(mars: pd.DataFrame) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    out: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for key, grp in mars.groupby(["sol", "sclk_int", "seqid", "point"], sort=False):
        g = grp.sort_values("wavenumber_cm_inv")
        out[key] = (
            g["wavenumber_cm_inv"].to_numpy(dtype=float),
            g["intensity_normalized_mean"].to_numpy(dtype=float),
        )
    return out


def select_named(supp: pd.DataFrame) -> pd.DataFrame:
    named = supp[supp["pass_label"].isin(["passes", "partial"])].copy()
    return named.sort_values(
        ["compound", "pass_label", "sol", "point_number", "gdp_signed_offset_cm"],
        kind="stable",
    ).reset_index(drop=True)


def make_figure(
    row, wn: np.ndarray, y: np.ndarray,
    rs_obs_compound: pd.DataFrame, catalog_compound: pd.DataFrame,
) -> plt.Figure:
    color = COMPOUND_COLOR[row.compound]
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.18)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    gdp_catalog = float(row.gdp_peak_center)

    ax_top.axvspan(
        gdp_catalog - COSINE_HALF_WINDOW, gdp_catalog + COSINE_HALF_WINDOW,
        color=color, alpha=0.06, lw=0,
    )
    ax_top.axvspan(
        gdp_catalog - R3_HALF_WINDOW, gdp_catalog + R3_HALF_WINDOW,
        color=color, alpha=0.18, lw=0,
    )

    ax_top.plot(wn, y, color=color, lw=0.8)

    a_cat = catalog_compound[catalog_compound["class_label"] == "A"]
    for _, p in a_cat.iterrows():
        pos = float(p["position_cm_inv"])
        is_gdp = abs(pos - gdp_catalog) < 1e-3
        ax_top.axvline(
            pos, color=color,
            lw=1.8 if is_gdp else 0.9,
            alpha=0.85 if is_gdp else 0.55,
        )
    b_cat = catalog_compound[catalog_compound["class_label"] == "B"]
    for _, p in b_cat.iterrows():
        ax_top.axvline(float(p["position_cm_inv"]), color=B_PEAK_COLOR, lw=0.5, alpha=0.55)

    a_score = rs_obs_compound[rs_obs_compound["peak_class"] == "A"]
    for _, rs in a_score.iterrows():
        wlo, whi = float(rs["local_window_lo"]), float(rs["local_window_hi"])
        center = float(rs["peak_center_catalog"])
        xs = np.linspace(wlo, whi, 200)
        xc = xs - center
        trend = (
            float(rs["poly_c0"])
            + float(rs["poly_c1"]) * xc
            + float(rs["poly_c2"]) * xc ** 2
        )
        ax_top.plot(xs, trend, color=color, lw=0.6, ls="--", alpha=0.55)

    ax_top.set_xlim(150, 1700)
    ax_top.set_ylabel("Intensity (normalized)")
    ax_top.set_title(
        f"{row.compound} | sol {int(row.sol)} pt {int(row.point_number)} "
        f"{row.target_name}"
    )
    plt.setp(ax_top.get_xticklabels(), visible=False)

    reason = row.failure_or_caveat_reason
    reason_line = (
        f"reason: {reason}\n"
        if isinstance(reason, str) and reason else ""
    )
    txt = (
        f"tier: {row.tier}\n"
        f"pass: {row.pass_label}\n"
        f"{reason_line}"
        f"gdp_snr: {row.gdp_snr:.2f}\n"
        f"gdp_offset: {row.gdp_signed_offset_cm:+.3f} cm-1\n"
        f"gdp_curvature: {int(row.gdp_curvature_sign):+d}\n"
        f"cation: {int(row.n_cation_peaks_curvature_clean)}/"
        f"{int(row.n_cation_peaks_total)} = "
        f"{row.cation_corroboration_fraction:.2f}\n"
        f"cos_pct: {row.cosine_percentile_within_compound_full_corpus:.1f}\n"
        f"qflags: {int(row.n_quality_flags)}"
    )
    # Annotation rendered as fig.text below the bottom panel — see end of
    # function. Keeping `txt` defined here because it references `row` locals.

    for _, rs in a_score.iterrows():
        wlo, whi = float(rs["local_window_lo"]), float(rs["local_window_hi"])
        center = float(rs["peak_center_catalog"])
        mask = (wn >= wlo) & (wn <= whi)
        if not mask.any():
            continue
        ww, yy = wn[mask], y[mask]
        xc = ww - center
        trend = (
            float(rs["poly_c0"])
            + float(rs["poly_c1"]) * xc
            + float(rs["poly_c2"]) * xc ** 2
        )
        det = yy - trend
        ax_bot.plot(ww, det, color=color, lw=0.8)
        sigma = float(rs["local_mad_detrended"])
        is_gdp = abs(center - gdp_catalog) < 1e-3
        if np.isfinite(sigma) and sigma > 0:
            ax_bot.plot([wlo, whi], [3 * sigma, 3 * sigma], color="#888888", ls="--", lw=0.6)
            ax_bot.plot([wlo, whi], [5 * sigma, 5 * sigma], color="#444444", ls="--", lw=0.6)
            if is_gdp:
                ax_bot.text(
                    whi, 3 * sigma, r"  3$\sigma$",
                    color="#888888", fontsize=6, va="center", ha="left",
                )
                ax_bot.text(
                    whi, 5 * sigma, r"  5$\sigma$",
                    color="#444444", fontsize=6, va="center", ha="left",
                )
        if is_gdp:
            argmax = float(rs["peak_max_position"])
            ax_bot.axvline(argmax, color=color, lw=1.0, alpha=0.85)
            ax_bot.plot(argmax, 0, marker="v", color=color, ms=7, clip_on=False)

    ax_bot.axhline(0, color="black", lw=0.4)
    ax_bot.set_xlim(150, 1700)
    ax_bot.set_xlabel("Raman shift (cm$^{-1}$)")
    ax_bot.set_ylabel("Detrended residual")

    # Reserve bottom 20% of the figure for the annotation block; fig.text
    # is layout-unaware so we set the margin explicitly. tight_layout
    # dropped (it warned on this gridspec/sharex configuration in the F.1
    # smoke test) — explicit subplots_adjust gives the same result without
    # the warning and without the constrained_layout/fig.text interaction
    # quirk.
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.20, hspace=0.18)
    fig.text(
        0.05, 0.02, txt, ha="left", va="bottom",
        fontsize=8, family="monospace",
        bbox=dict(facecolor="white", edgecolor="lightgrey", boxstyle="round,pad=0.4"),
    )
    return fig


def filename_for(row) -> str:
    return (
        f"{COMPOUND_SHORT[row.compound]}_"
        f"sol{int(row.sol):04d}_"
        f"sclk{int(row.sclk_int)}_"
        f"pt{int(row.point_number):02d}_"
        f"{safe_name(row.target_name)}.pdf"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--limit", type=int, default=None,
        help="If set, only plot the first N rows of the sorted named-candidate "
             "set (smoke test). Default: full run (475 panels).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    supp, raw, cat, mars = load_inputs()
    spectra = index_spectra(mars)
    named = select_named(supp)
    if args.limit is not None:
        named = named.head(args.limit).reset_index(drop=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ("headlines", "passes", "partials", "combined"):
        (OUT_ROOT / sub).mkdir(exist_ok=True)

    raw_g = raw.groupby(
        ["sol", "sclk_int", "seqid", "point_number", "compound"], sort=False,
    )
    cat_g = cat.groupby("compound", sort=False)

    combined_pdfs: dict[Path, PdfPages] = {}

    def combined_path(compound: str, pass_label: str) -> Path:
        # Normalize the parquet's singular "partial" pass_label to plural
        # "partials" so the combined-PDF filename matches the partials/
        # bucket directory.
        bucket = "partials" if pass_label == "partial" else pass_label
        return OUT_ROOT / "combined" / f"combined_{bucket}_{COMPOUND_SHORT[compound]}.pdf"

    index_rows: list[dict] = []
    n_panels = len(named)
    print(f"Plotting {n_panels} panels...", file=sys.stderr)

    for i, row in enumerate(named.itertuples(index=False), 1):
        spec_key = (int(row.sol), int(row.sclk_int), row.seqid, int(row.point_number))
        wn, y = spectra[spec_key]
        rs_subset = raw_g.get_group(spec_key + (row.compound,))
        cat_subset = cat_g.get_group(row.compound)

        fig = make_figure(row, wn, y, rs_subset, cat_subset)

        is_headline = (
            row.compound, int(row.sol), int(row.sclk_int), int(row.point_number)
        ) in HEADLINES
        bucket = "passes" if row.pass_label == "passes" else "partials"
        fname = filename_for(row)
        primary_path = OUT_ROOT / bucket / fname
        fig.savefig(primary_path, format="pdf")
        if is_headline:
            fig.savefig(OUT_ROOT / "headlines" / fname, format="pdf")

        cpath = combined_path(row.compound, row.pass_label)
        if cpath not in combined_pdfs:
            combined_pdfs[cpath] = PdfPages(cpath)
        combined_pdfs[cpath].savefig(fig)

        plt.close(fig)

        index_rows.append({
            "relative_path": str(primary_path.relative_to(OUT_ROOT)),
            "compound": row.compound,
            "sol": int(row.sol),
            "sclk_int": int(row.sclk_int),
            "seqid": row.seqid,
            "point_number": int(row.point_number),
            "target_name": row.target_name,
            "target_classification": row.target_classification,
            "tier": row.tier,
            "pass_label": row.pass_label,
            "failure_or_caveat_reason": row.failure_or_caveat_reason,
            "gdp_snr": float(row.gdp_snr),
            "gdp_signed_offset_cm": float(row.gdp_signed_offset_cm),
            "gdp_curvature_sign": int(row.gdp_curvature_sign),
            "cosine_pct": float(row.cosine_percentile_within_compound_full_corpus),
            "cation_corroboration_fraction": float(row.cation_corroboration_fraction),
            "n_quality_flags": int(row.n_quality_flags),
            "is_headline": bool(is_headline),
        })

        if i % 50 == 0 or i == n_panels:
            print(f"  {i}/{n_panels}", file=sys.stderr)

    for pdf in combined_pdfs.values():
        pdf.close()

    pd.DataFrame(index_rows).to_csv(OUT_ROOT / "INDEX.csv", index=False)
    n_headline_copies = sum(1 for r in index_rows if r["is_headline"])
    print(
        f"Done. {len(index_rows)} primary PDFs + {n_headline_copies} headline copies + "
        f"{len(combined_pdfs)} combined PDFs + INDEX.csv "
        f"under {OUT_ROOT.relative_to(REPO)}/",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
