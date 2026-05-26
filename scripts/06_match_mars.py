"""Match Mars spectra against the lab reference catalog.

Reads `data/mars_processed/mars_processed.parquet` and the reference library
parquets, computes per-peak SNR via local polynomial detrending, per-peak
quality flags, per-(spectrum x compound) cosine in compound-specific
diagnostic windows, and a per-spectrum bump-region structure indicator.
Writes three parquets in `results/`.

Confound-blind and tier-blind by design. Tier logic, evidence_pattern,
peak quality flag aggregation, cosine percentile ranking, and confound
annotation all live in 07. ARTIFACT-class peaks are scored normally; their
exclusion from tier logic happens in 07.

Peer-peak fit-exclusion is applied cross-compound: every catalog peak whose
tolerance window overlaps the local fit window is excluded, regardless of
compound. Handoff v4 text on Q1 step 2 was ambiguous on this point; the
script settles it as cross-compound. The same rule applies inside
compute_cosine_in_diagnostic_windows.

References:
  Sturm et al. 2023 (PMC10612323). Peak quality flags as a generic-Raman
  best practice.
  See docs/figures/d_design/D_handoff_v4.md Q1, Q2, Q3, Q4, Q6 for the
  design specification.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _processing import (  # noqa: E402
    compute_cosine_in_diagnostic_windows,
    compute_local_polynomial_detrending,
    compute_peak_quality_flags,
)

# Inputs.
MARS_PATH = REPO_ROOT / "data" / "mars_processed" / "mars_processed.parquet"
CATALOG_PATH = REPO_ROOT / "data" / "reference_library" / "peak_catalog.parquet"
REF_SPECTRA_PATH = (
    REPO_ROOT / "data" / "reference_library" / "reference_spectra.parquet"
)

# Outputs (results/ directory is gitignored).
RESULTS_DIR = REPO_ROOT / "results"
RAW_SCORES_PATH = RESULTS_DIR / "raw_scores.parquet"
COSINE_PATH = RESULTS_DIR / "cosine_diagnostic.parquet"
BUMP_PATH = RESULTS_DIR / "bump_indicator.parquet"
PROVENANCE_PATH = RESULTS_DIR / ".provenance.json"

COMPOUND_ORDER = (
    "EMIM-FeCl4",
    "EMIM-FeBr4",
    "EMIM2-Fe2Cl7",
    "EMIM2-FeSO4",
    # EMIM-cation: shared cation peaks moved here by the post-audit
    # 05 rewrite. Scored per shot so that 12 can run cation
    # corroboration cross-anion-compound. Cosine for this compound is
    # computed but is not used downstream.
    "EMIM-cation",
)

# Per-rule tolerance overrides keyed by (compound, rounded center). Mirrors
# the four FeSO4 doublet overrides applied in 05 so the per-peak
# tolerance_used column in raw_scores reflects the windows 05 used for class
# assignment. The catalog stores tolerance_default_cm (10.0) and
# tolerance_tight_cm (5.0); 06 uses the tight value as the default and the
# overrides below for the four doublets.
TOLERANCE_OVERRIDES: dict[tuple[str, float], float] = {
    ("EMIM2-FeSO4", 1022.01): 2.5,
    ("EMIM2-FeSO4", 1027.38): 2.5,
    ("EMIM2-FeSO4", 1086.60): 2.5,
    ("EMIM2-FeSO4", 1091.81): 2.5,
}
DEFAULT_TOLERANCE = 5.0

# Design constants for the SNR primitive.
POLY_DEGREE = 2
HALF_WINDOW = 30.0
PEER_TOL = 5.0

# Cosine-window half-width.
COSINE_HALF_WINDOW = 15.0

# Bump-region structure indicator (Q3): mean intensity in [BUMP_RANGE_LO,
# BUMP_RANGE_HI] divided by mean intensity in [BUMP_REF_LO, BUMP_REF_HI].
BUMP_RANGE_LO = 200.0
BUMP_RANGE_HI = 530.0
BUMP_REF_LO = 600.0
BUMP_REF_HI = 900.0


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def tolerance_for(compound: str, center: float) -> float:
    """Return the matcher tolerance for one catalog peak.

    Default 5.0 cm-1 (the catalog `tolerance_tight_cm` value). 2.5 cm-1 for
    the four FeSO4 doublet rules listed in TOLERANCE_OVERRIDES.
    """
    for (cmp, ctr), v in TOLERANCE_OVERRIDES.items():
        if cmp == compound and abs(ctr - center) < 0.05:
            return v
    return DEFAULT_TOLERANCE


def load_catalog() -> pd.DataFrame:
    """Load the 46 aggregate-convolved catalog rows.

    Adds two derived columns:
      tolerance_used   per-peak matcher tolerance (5.0 / 2.5)
      peak_subclass    'primary' / 'secondary' / '' parsed from class_notes
                       prefix for class A peaks; empty for B / C / ARTIFACT.
    """
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
    df = df.sort_values(["compound", "position_cm_inv"]).reset_index(drop=True)
    return df


def load_reference_means(
    compound_order: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-compound mean of intensity_normalized_convolved across acquisitions
    flagged usable_for_supercam=True. Returns {compound: (wavenumbers, mean)}.
    """
    df = pd.read_parquet(REF_SPECTRA_PATH)
    df = df[df["usable_for_supercam"] == True].copy()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in compound_order:
        sub = df[df["compound"] == c]
        agg = (
            sub.groupby("wavenumber_cm_inv")["intensity_normalized_convolved"]
            .mean()
            .reset_index()
            .sort_values("wavenumber_cm_inv")
        )
        out[c] = (
            agg["wavenumber_cm_inv"].to_numpy(dtype=float),
            agg["intensity_normalized_convolved"].to_numpy(dtype=float),
        )
    return out


def build_peer_centers_for_window(
    catalog: pd.DataFrame, target_center: float, half_window: float = HALF_WINDOW
) -> list[float]:
    """Catalog centers whose tolerance window overlaps [target_center +/-
    half_window], excluding the target itself. Cross-compound; uses each
    peer's own tolerance_used for the overlap check.
    """
    W_lo = target_center - half_window
    W_hi = target_center + half_window
    out = []
    for c, tol in zip(catalog["position_cm_inv"], catalog["tolerance_used"]):
        c = float(c)
        tol = float(tol)
        if c + tol < W_lo or c - tol > W_hi:
            continue
        if abs(c - target_center) < 1e-3:
            continue
        out.append(c)
    return out


def build_observation_index(mars: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sol",
        "sclk_int",
        "seqid",
        "point",
        "target_name",
        "target_classification",
    ]
    obs = (
        mars[cols]
        .drop_duplicates()
        .sort_values(["sol", "sclk_int", "seqid", "point"])
        .reset_index(drop=True)
    )
    return obs


def group_mars_by_observation(
    mars: pd.DataFrame,
) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
    """Pre-compute per-observation (wavenumbers, intensities) arrays with
    NaN-edge points dropped. Keyed by (sol, sclk_int, seqid, point).
    """
    sub = mars.sort_values(
        ["sol", "sclk_int", "seqid", "point", "wavenumber_cm_inv"]
    )
    out: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for keys, grp in sub.groupby(
        ["sol", "sclk_int", "seqid", "point"], sort=False
    ):
        wn = grp["wavenumber_cm_inv"].to_numpy(dtype=float)
        y = grp["intensity_normalized_mean"].to_numpy(dtype=float)
        finite = np.isfinite(y)
        norm_key = (int(keys[0]), int(keys[1]), str(keys[2]), int(keys[3]))
        out[norm_key] = (wn[finite], y[finite])
    return out


def main() -> None:
    if not MARS_PATH.is_file():
        raise FileNotFoundError(f"Missing input: {MARS_PATH}")
    if not CATALOG_PATH.is_file():
        raise FileNotFoundError(f"Missing input: {CATALOG_PATH}")
    if not REF_SPECTRA_PATH.is_file():
        raise FileNotFoundError(f"Missing input: {REF_SPECTRA_PATH}")
    RESULTS_DIR.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    mars = pd.read_parquet(MARS_PATH)
    catalog = load_catalog()
    # Post-audit catalog row count: 5 A-anion + 2 B-anion + 4 C-anion +
    # 22 EMIM-cation + 2 ARTIFACT = 35.
    if len(catalog) != 35:
        raise ValueError(
            f"expected 35 aggregate-convolved catalog rows, got {len(catalog)}"
        )
    ref_means = load_reference_means(COMPOUND_ORDER)
    obs_index = build_observation_index(mars)
    obs_data = group_mars_by_observation(mars)

    # All catalog centers serve as the global peer pool. For SNR we look up
    # per-peak peer lists below; for cosine we pass the full pool and let the
    # primitive filter per window.
    all_catalog_centers = [float(c) for c in catalog["position_cm_inv"]]

    # Per-peak peer lists, computed once from the catalog (catalog is global
    # to 06; doesn't depend on the Mars observation).
    peer_lists: dict[int, list[float]] = {}
    for idx, row in catalog.iterrows():
        peer_lists[int(idx)] = build_peer_centers_for_window(
            catalog, float(row["position_cm_inv"])
        )

    # Cosine window centers per compound: A primary + secondary peaks, sorted
    # ascending. Built once from the catalog.
    cosine_windows: dict[str, list[float]] = {}
    for c in COMPOUND_ORDER:
        a = catalog[
            (catalog["compound"] == c) & (catalog["class_label"] == "A")
        ]
        cosine_windows[c] = sorted(float(p) for p in a["position_cm_inv"])

    raw_rows: list[dict] = []
    cosine_rows: list[dict] = []
    bump_rows: list[dict] = []

    for _, obs in obs_index.iterrows():
        key = (
            int(obs["sol"]),
            int(obs["sclk_int"]),
            str(obs["seqid"]),
            int(obs["point"]),
        )
        wn, y = obs_data[key]

        # Bump indicator (Q3).
        bump_mask = (wn >= BUMP_RANGE_LO) & (wn <= BUMP_RANGE_HI)
        ref_mask = (wn >= BUMP_REF_LO) & (wn <= BUMP_REF_HI)
        bump_num = float(np.mean(y[bump_mask])) if bump_mask.any() else float("nan")
        bump_den = float(np.mean(y[ref_mask])) if ref_mask.any() else float("nan")
        if not np.isfinite(bump_den) or bump_den == 0.0:
            bump_indicator = float("nan")
        else:
            bump_indicator = bump_num / bump_den
        bump_rows.append(
            {
                "sol": key[0],
                "sclk_int": key[1],
                "seqid": key[2],
                "point_number": key[3],
                "bump_region_structure_indicator": bump_indicator,
            }
        )

        target_name = (
            "" if pd.isna(obs["target_name"]) else str(obs["target_name"])
        )
        target_class = (
            ""
            if pd.isna(obs["target_classification"])
            else str(obs["target_classification"])
        )

        for compound in COMPOUND_ORDER:
            cat_sub = catalog[catalog["compound"] == compound].sort_values(
                "position_cm_inv"
            )
            for cat_idx, peak in cat_sub.iterrows():
                target_c = float(peak["position_cm_inv"])
                target_tol = float(peak["tolerance_used"])
                peers = peer_lists[int(cat_idx)]
                result = compute_local_polynomial_detrending(
                    wn,
                    y,
                    target_c,
                    target_tol,
                    peers,
                    half_window=HALF_WINDOW,
                    peer_tol=PEER_TOL,
                    poly_degree=POLY_DEGREE,
                )
                flags = compute_peak_quality_flags(
                    result["detrended_window"],
                    result["wavenumbers_window"],
                    result["peak_position"],
                    target_c,
                    target_tol,
                )
                c0, c1, c2 = result["poly_coeffs"]
                raw_rows.append(
                    {
                        "sol": key[0],
                        "sclk_int": key[1],
                        "seqid": key[2],
                        "point_number": key[3],
                        "target_name": target_name,
                        "target_classification": target_class,
                        "compound": compound,
                        "peak_center_catalog": target_c,
                        "peak_class": ""
                        if pd.isna(peak["class_label"])
                        else str(peak["class_label"]),
                        "peak_subclass": str(peak["peak_subclass"]),
                        "fiber_bump_overlap": bool(peak["fiber_bump_overlap"]),
                        "peak_height_detrended": result["peak_height"],
                        "peak_max_position": result["peak_position"],
                        "local_mad_detrended": result["local_mad"],
                        "snr_detrended": result["snr"],
                        "position_offset_normalized": flags[
                            "position_offset_normalized"
                        ],
                        "peak_curvature_sign": flags["peak_curvature_sign"],
                        "poly_c0": c0,
                        "poly_c1": c1,
                        "poly_c2": c2,
                        "fit_n_points": result["n_fit_points"],
                        "local_window_lo": result["window_lo"],
                        "local_window_hi": result["window_hi"],
                        "tolerance_used": target_tol,
                    }
                )

            # Cosine: align Mars and reference on a common wavenumber grid (both
            # are on the integer 150-1700 grid, so intersect1d is direct).
            ref_wn, ref_y = ref_means[compound]
            common_wn, m_idx, r_idx = np.intersect1d(
                wn, ref_wn, return_indices=True
            )
            if common_wn.size == 0:
                cosine_val = float("nan")
            else:
                cosine_val = compute_cosine_in_diagnostic_windows(
                    y[m_idx],
                    ref_y[r_idx],
                    common_wn,
                    cosine_windows[compound],
                    peer_peak_centers=all_catalog_centers,
                    tolerance=DEFAULT_TOLERANCE,
                    half_window=COSINE_HALF_WINDOW,
                    peer_tol=PEER_TOL,
                    poly_degree=POLY_DEGREE,
                )
            cosine_rows.append(
                {
                    "sol": key[0],
                    "sclk_int": key[1],
                    "seqid": key[2],
                    "point_number": key[3],
                    "compound": compound,
                    "cosine_diagnostic": cosine_val,
                }
            )

    # Build dataframes, sort, write parquets.
    raw_scores_df = (
        pd.DataFrame(raw_rows)
        .sort_values(
            [
                "sol",
                "sclk_int",
                "seqid",
                "point_number",
                "compound",
                "peak_center_catalog",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    cosine_df = (
        pd.DataFrame(cosine_rows)
        .sort_values(
            ["sol", "sclk_int", "seqid", "point_number", "compound"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    bump_df = (
        pd.DataFrame(bump_rows)
        .sort_values(
            ["sol", "sclk_int", "seqid", "point_number"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    pq.write_table(
        pa.Table.from_pandas(raw_scores_df, preserve_index=False),
        RAW_SCORES_PATH,
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pandas(cosine_df, preserve_index=False),
        COSINE_PATH,
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pandas(bump_df, preserve_index=False),
        BUMP_PATH,
        compression="zstd",
    )

    elapsed = time.perf_counter() - t0
    provenance = {
        "script_path": str(Path("scripts") / Path(__file__).name),
        "git_commit_hash": git_head(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_input_observations": int(len(obs_index)),
        "n_input_catalog_peaks": int(len(catalog)),
        "n_output_rows": {
            "raw_scores.parquet": int(len(raw_scores_df)),
            "cosine_diagnostic.parquet": int(len(cosine_df)),
            "bump_indicator.parquet": int(len(bump_df)),
        },
        "design_constants": {
            "POLY_DEGREE": POLY_DEGREE,
            "HALF_WINDOW": HALF_WINDOW,
            "PEER_TOL": PEER_TOL,
            "COSINE_HALF_WINDOW": COSINE_HALF_WINDOW,
            "BUMP_RANGE_LO": BUMP_RANGE_LO,
            "BUMP_RANGE_HI": BUMP_RANGE_HI,
            "BUMP_REF_LO": BUMP_REF_LO,
            "BUMP_REF_HI": BUMP_REF_HI,
            "COMPOUND_ORDER": list(COMPOUND_ORDER),
            "cosine_window_centers": {
                c: cosine_windows[c] for c in COMPOUND_ORDER
            },
        },
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wall clock: {elapsed:.2f} s")
    print(f"Wrote {RAW_SCORES_PATH.name}: {len(raw_scores_df)} rows")
    print(f"Wrote {COSINE_PATH.name}: {len(cosine_df)} rows")
    print(f"Wrote {BUMP_PATH.name}: {len(bump_df)} rows")


if __name__ == "__main__":
    main()
