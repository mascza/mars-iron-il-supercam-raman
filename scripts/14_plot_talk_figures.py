"""Talk-figure panels: one PDF per top-tier candidate, single panel per
candidate (no residual subplot), Wong-palette color per compound, anion
catalog peaks vlined with numeric position labels at base, cation peaks
overlaid as lighter dashed vlines, fiber-bump (200-530 cm-1) shaded.

Selection criteria (approved):
  FeCl4   : n_anion_peaks_curvature_clean == 2 AND cation_corroboration_fraction >= 0.4
  FeBr4   : n_anion_peaks_curvature_clean == 4
            OR (n_anion_peaks_curvature_clean == 3 AND cation_corroboration_fraction >= 1.0)
  Fe2Cl7  : n_anion_peaks_curvature_clean == 3
  FeSO4   : cation_corroboration_fraction >= 0.25 AND n_quality_flags == 0,
            top 8 by gdp_snr desc (R5 structural lockout — single A peak)

Where n_anion_peaks_curvature_clean = count of catalog A peaks (primary +
secondary) with snr_detrended >= 3.0 AND peak_curvature_sign == -1.

Reads (no writes outside results/figures/talk_panels/):
  results/strong_and_candidate_classification.parquet
  results/raw_scores.parquet
  data/reference_library/peak_catalog.parquet
  data/mars_processed/mars_processed.parquet

Writes:
  results/figures/talk_panels/FeCl4/      *.pdf
  results/figures/talk_panels/FeBr4/      *.pdf
  results/figures/talk_panels/Fe2Cl7/     *.pdf
  results/figures/talk_panels/FeSO4/      *.pdf
  results/figures/talk_panels/INDEX.csv

Filename: {COMPOUND_SHORT}_sol{NNNN}_sclk{N}_pt{NN}_{target_safe}.pdf
(same convention as 13_plot_candidates.py).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

REPO = Path(__file__).resolve().parent.parent
SUPP_PATH = REPO / "results" / "strong_and_candidate_classification.parquet"
RAW_PATH = REPO / "results" / "raw_scores.parquet"
CATALOG_PATH = REPO / "data" / "reference_library" / "peak_catalog.parquet"
MARS_PATH = REPO / "data" / "mars_processed" / "mars_processed.parquet"
OUT_ROOT = REPO / "results" / "figures" / "talk_panels"

COMPOUND_COLOR = {
    "EMIM-FeCl4": "#0072B2",
    "EMIM-FeBr4": "#E69F00",
    "EMIM2-Fe2Cl7": "#CC79A7",
    "EMIM2-FeSO4": "#009E73",
}
COMPOUND_SHORT = {
    "EMIM-FeCl4": "FeCl4",
    "EMIM-FeBr4": "FeBr4",
    "EMIM2-Fe2Cl7": "Fe2Cl7",
    "EMIM2-FeSO4": "FeSO4",
}
COMPOUND_FULL = {
    "EMIM-FeCl4": "[EMIM][FeCl₄]",
    "EMIM-FeBr4": "[EMIM][FeBr₄]",
    "EMIM2-Fe2Cl7": "[EMIM]₂[Fe₂Cl₇]",
    "EMIM2-FeSO4": "[EMIM]₂[Fe(SO₄)₂]",
}
SHORT_TO_FULL = {v: k for k, v in COMPOUND_SHORT.items()}

# X range per compound (inclusive). Anion peaks plus ~50 cm-1 margin and
# enough context for the closest cation peaks.
X_RANGE = {
    "EMIM-FeCl4": (150.0, 700.0),
    "EMIM-FeBr4": (150.0, 700.0),
    "EMIM2-Fe2Cl7": (150.0, 500.0),
    "EMIM2-FeSO4": (850.0, 1100.0),
}

FIBER_BUMP_LO = 200.0
FIBER_BUMP_HI = 530.0
MARS_COLOR = "#2B2B2B"
FIBER_BUMP_COLOR = "#D0D0D0"


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def safe_name(s: str, max_len: int = 32) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return (s[:max_len] or "unknown").rstrip("_")


def lighter(color_hex: str, blend: float = 0.55) -> tuple[float, float, float]:
    """Blend the color toward white. blend=0 returns the color, blend=1 white."""
    rgb = np.array(mcolors.to_rgb(color_hex))
    return tuple(rgb + (1.0 - rgb) * blend)


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


def add_n_anion_clean(supp: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["sol", "sclk_int", "seqid", "point_number", "compound"]
    a = raw[raw["peak_class"] == "A"].copy()
    a["anion_clean"] = (a["snr_detrended"] >= 3.0) & (a["peak_curvature_sign"] == -1)
    agg = (
        a.groupby(keys, as_index=False)["anion_clean"]
        .sum()
        .rename(columns={"anion_clean": "n_anion_peaks_curvature_clean"})
    )
    n_total = (
        a.groupby(keys, as_index=False)["peak_class"]
        .size()
        .rename(columns={"size": "n_anion_peaks_total"})
    )
    out = supp.merge(agg, on=keys, how="left").merge(n_total, on=keys, how="left")
    out["n_anion_peaks_curvature_clean"] = (
        out["n_anion_peaks_curvature_clean"].fillna(0).astype(int)
    )
    out["n_anion_peaks_total"] = out["n_anion_peaks_total"].fillna(0).astype(int)
    return out


def select_for_compound(supp_aug: pd.DataFrame, compound: str) -> pd.DataFrame:
    df = supp_aug[
        (supp_aug["compound"] == compound)
        & (supp_aug["pass_label"].isin(["passes", "partial"]))
    ].copy()
    if compound == "EMIM-FeCl4":
        sel = df[
            (df["n_anion_peaks_curvature_clean"] == 2)
            & (df["cation_corroboration_fraction"] >= 0.4)
        ]
        sel = sel.sort_values(
            ["n_anion_peaks_curvature_clean", "cation_corroboration_fraction",
             "gdp_snr", "sol", "sclk_int"],
            ascending=[False, False, False, True, True],
            kind="stable",
        )
    elif compound == "EMIM-FeBr4":
        sel = df[
            (df["n_anion_peaks_curvature_clean"] == 4)
            | (
                (df["n_anion_peaks_curvature_clean"] == 3)
                & (df["cation_corroboration_fraction"] >= 1.0)
            )
        ]
        sel = sel.sort_values(
            ["n_anion_peaks_curvature_clean", "cation_corroboration_fraction",
             "gdp_snr", "sol", "sclk_int"],
            ascending=[False, False, False, True, True],
            kind="stable",
        )
    elif compound == "EMIM2-Fe2Cl7":
        sel = df[df["n_anion_peaks_curvature_clean"] == 3]
        sel = sel.sort_values(
            ["n_anion_peaks_curvature_clean", "cation_corroboration_fraction",
             "gdp_snr", "sol", "sclk_int"],
            ascending=[False, False, False, True, True],
            kind="stable",
        )
    elif compound == "EMIM2-FeSO4":
        sel = df[
            (df["cation_corroboration_fraction"] >= 0.25)
            & (df["n_quality_flags"] == 0)
        ]
        sel = sel.sort_values(
            ["gdp_snr", "sol", "sclk_int"],
            ascending=[False, True, True],
            kind="stable",
        ).head(8)
    else:
        raise ValueError(f"unknown compound: {compound}")
    return sel.reset_index(drop=True)


def make_figure(
    row, wn: np.ndarray, y: np.ndarray, catalog_compound: pd.DataFrame,
) -> plt.Figure:
    color = COMPOUND_COLOR[row.compound]
    color_light = lighter(color, blend=0.55)
    short = COMPOUND_SHORT[row.compound]
    xlo, xhi = X_RANGE[row.compound]

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)

    ax.axvspan(
        max(FIBER_BUMP_LO, xlo), min(FIBER_BUMP_HI, xhi),
        color=FIBER_BUMP_COLOR, alpha=0.35, lw=0,
    )

    mask = (wn >= xlo) & (wn <= xhi)
    ax.plot(wn[mask], y[mask], color=MARS_COLOR, lw=2.0)

    a_cat = catalog_compound[catalog_compound["class_label"] == "A"]
    b_cat = catalog_compound[catalog_compound["class_label"] == "B"]

    a_in = a_cat[
        (a_cat["position_cm_inv"] >= xlo) & (a_cat["position_cm_inv"] <= xhi)
    ]
    b_in = b_cat[
        (b_cat["position_cm_inv"] >= xlo) & (b_cat["position_cm_inv"] <= xhi)
    ]

    for _, p in a_in.iterrows():
        ax.axvline(float(p["position_cm_inv"]), color=color, lw=3.0, alpha=0.85)
    for _, p in b_in.iterrows():
        ax.axvline(
            float(p["position_cm_inv"]),
            color=color_light, lw=1.5, alpha=1.0, ls="--",
        )

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for _, p in a_in.iterrows():
        pos = float(p["position_cm_inv"])
        ax.text(
            pos, 0.02, f"{int(round(pos))}",
            transform=trans, fontsize=11, color=color,
            ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=2),
        )

    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=16)
    ax.set_ylabel("Intensity (normalized)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)

    n_anion = int(row.n_anion_peaks_curvature_clean)
    n_total = int(row.n_anion_peaks_total)
    full = COMPOUND_FULL[row.compound]
    title = (
        f"{full} | sol {int(row.sol)} | {row.target_name} | "
        f"pt {int(row.point_number)} | "
        f"{n_anion}/{n_total} anion peaks above 3σ"
    )
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.97)
    fig.text(
        0.5, 0.92,
        "Mars calibrated data: SuperCam (Wiens et al. 2021; "
        "Maurice et al. 2021), Mars 2020 Perseverance",
        ha="center", va="top", fontsize=10, color="#666666",
    )

    legend_handles = [
        Line2D([0], [0], color=MARS_COLOR, lw=2.0, label="Mars spectrum"),
        Line2D([0], [0], color=color, lw=3.0, alpha=0.85,
               label=f"{full} anion lab markers"),
        Line2D([0], [0], color=color_light, lw=1.5, ls="--",
               label=f"{full} cation lab markers"),
        mpatches.Patch(facecolor=FIBER_BUMP_COLOR, alpha=0.35,
                       label="Fiber-bump uncertainty region"),
    ]
    leg = fig.legend(
        handles=legend_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.05),
        ncol=4, frameon=True, fontsize=13,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor(color)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.20)
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
        "--compound", action="append", default=None,
        choices=["FeCl4", "FeBr4", "Fe2Cl7", "FeSO4"],
        help="Compound short name (repeatable). Default: all four.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="If set, only plot the first N rows per selected compound (smoke test).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()

    if args.compound:
        compounds = [SHORT_TO_FULL[s] for s in args.compound]
    else:
        compounds = list(COMPOUND_COLOR.keys())

    supp, raw, cat, mars = load_inputs()
    spectra = index_spectra(mars)
    supp_aug = add_n_anion_clean(supp, raw)
    cat_g = cat.groupby("compound", sort=False)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for short in COMPOUND_SHORT.values():
        (OUT_ROOT / short).mkdir(exist_ok=True)

    index_rows: list[dict] = []
    total_panels = 0
    for compound in compounds:
        sel = select_for_compound(supp_aug, compound)
        if args.limit is not None:
            sel = sel.head(args.limit).reset_index(drop=True)
        short = COMPOUND_SHORT[compound]
        print(f"{short}: {len(sel)} panels", file=sys.stderr)

        cat_subset = cat_g.get_group(compound)

        for row in sel.itertuples(index=False):
            spec_key = (
                int(row.sol), int(row.sclk_int), row.seqid, int(row.point_number),
            )
            wn, y = spectra[spec_key]
            fig = make_figure(row, wn, y, cat_subset)
            fname = filename_for(row)
            out_path = OUT_ROOT / short / fname
            fig.savefig(out_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            total_panels += 1
            index_rows.append({
                "relative_path": str(out_path.relative_to(OUT_ROOT)),
                "compound": row.compound,
                "compound_short": short,
                "sol": int(row.sol),
                "sclk_int": int(row.sclk_int),
                "seqid": row.seqid,
                "point_number": int(row.point_number),
                "target_name": row.target_name,
                "target_classification": row.target_classification,
                "tier": row.tier,
                "pass_label": row.pass_label,
                "n_anion_peaks_curvature_clean": int(row.n_anion_peaks_curvature_clean),
                "n_anion_peaks_total": int(row.n_anion_peaks_total),
                "cation_corroboration_fraction": float(row.cation_corroboration_fraction),
                "gdp_snr": float(row.gdp_snr),
                "gdp_signed_offset_cm": float(row.gdp_signed_offset_cm),
                "cosine_pct": float(row.cosine_percentile_within_compound_full_corpus),
                "n_quality_flags": int(row.n_quality_flags),
            })

    pd.DataFrame(index_rows).to_csv(OUT_ROOT / "INDEX.csv", index=False)
    print(
        f"Done. {total_panels} PDFs + INDEX.csv under "
        f"{OUT_ROOT.relative_to(REPO)}/",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
