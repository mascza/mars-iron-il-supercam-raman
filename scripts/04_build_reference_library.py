"""Build the SuperCam-resolution reference library from lab spectra.

Reads `data/lab_processed/lab_processed.parquet`, convolves each spectrum to
SuperCam's documented Raman resolution (12 cm-1 FWHM, User Guide §2.2),
fits Lorentzian profiles to peaks at both lab and convolved resolutions,
aggregates per compound, and writes a peak catalog plus a cross-compound
comparison summary.

References:
  Mars 2020 SuperCam Calibration and Data User Guide v5.0 §2.2 (Raman
    spectral resolution ~12 cm-1 FWHM).
  Gaussian convolution at FWHM follows the standard relation
    sigma = FWHM / (2 * sqrt(2 * ln 2)).
  Lorentzian profile: L(x) = A * gamma^2 / ((x - x0)^2 + gamma^2) + c,
    with HWHM = gamma, FWHM = 2 * gamma, area = pi * A * gamma.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _processing import (  # noqa: E402
    SUPERCAM_FWHM_CM,
    SUPERCAM_SIGMA_CM,
    gaussian_convolve,
    lorentzian,
)

GRID = np.arange(150.0, 1701.0, 1.0)

# find_peaks parameters. prominence=0.02 of normalized intensity rejects
# noise; distance=5 cm-1 prevents double-detecting shoulders on the 1 cm-1
# grid; width=2 requires the peak to span at least two grid points (rejects
# residual cosmic rays that ALS may have missed).
PEAK_FIND_PROMINENCE = 0.02
PEAK_FIND_DISTANCE = 5
PEAK_FIND_WIDTH = 2

# Lorentzian fit window and validity bounds.
FIT_WINDOW_HALF_CM = 10.0
FIT_X0_DRIFT_LIMIT_CM = 5.0
FIT_FWHM_LIMIT_CM = 100.0
FIT_BASELINE_LOWER = -0.05
FIT_BASELINE_UPPER = 0.1
# Reject fits whose r^2 is below this threshold. Filters bad fits (multi-peak
# windows, fits to noise) without filtering genuine low-intensity peaks.
FIT_R2_MIN = 0.85

# Aggregation parameters.
CLUSTER_TOLERANCE_CM = 5.0
PRESENCE_THRESHOLD = 0.5
TOLERANCE_DEFAULT_CM = 10.0
TOLERANCE_TIGHT_CM = 5.0
FIBER_BUMP_LO = 200.0
FIBER_BUMP_HI = 530.0
SUPERCAM_LASER_NM = 532

COMPOUND_ORDER = ("EMIM-FeCl4", "EMIM-FeBr4", "EMIM2-Fe2Cl7", "EMIM2-FeSO4")
LAB_OR_CONVOLVED = ("lab", "convolved")


def fit_one_peak(
    grid: np.ndarray,
    intensity: np.ndarray,
    candidate_idx: int,
    prominence: float,
    width_samples: float,
) -> tuple[dict | None, str | None]:
    """Fit a Lorentzian within +/-25 cm-1 of `candidate_idx`.

    Returns (peak_dict, None) on success or (None, reject_reason) on failure.
    `peak_dict` carries the fitted parameters plus diagnostics.
    """
    half = int(round(FIT_WINDOW_HALF_CM))
    lo = max(0, candidate_idx - half)
    hi = min(grid.size, candidate_idx + half + 1)
    x = grid[lo:hi]
    y = intensity[lo:hi]
    finite = np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = x.size
    if n < 5:
        return None, "fewer than 5 finite points in window"

    x0_guess = float(grid[candidate_idx])
    A_guess = float(intensity[candidate_idx]) - float(np.min(y))
    if A_guess <= 0.0:
        A_guess = float(intensity[candidate_idx])
    A_guess = float(np.clip(A_guess, 1e-3, 4.99))
    gamma_guess = max(0.5, float(width_samples) / 2.0)
    gamma_guess = min(gamma_guess, FIT_FWHM_LIMIT_CM / 2.0 - 0.1)
    c_guess = float(
        np.clip(np.min(y), FIT_BASELINE_LOWER + 1e-6, FIT_BASELINE_UPPER - 1e-6)
    )
    p0 = [x0_guess, gamma_guess, A_guess, c_guess]
    bounds = (
        [
            x0_guess - FIT_X0_DRIFT_LIMIT_CM,
            0.1,
            0.0,
            FIT_BASELINE_LOWER,
        ],
        [
            x0_guess + FIT_X0_DRIFT_LIMIT_CM,
            FIT_FWHM_LIMIT_CM / 2.0,
            5.0,
            FIT_BASELINE_UPPER,
        ],
    )
    try:
        popt, _ = curve_fit(lorentzian, x, y, p0=p0, bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError) as exc:
        return None, f"curve_fit failed ({type(exc).__name__})"

    x0, gamma, A, c = (float(v) for v in popt)
    fwhm = 2.0 * gamma
    if A <= 0.0:
        return None, "fitted amplitude not positive"
    if fwhm > FIT_FWHM_LIMIT_CM:
        return None, f"fitted FWHM {fwhm:.1f} > {FIT_FWHM_LIMIT_CM}"
    if abs(x0 - x0_guess) > FIT_X0_DRIFT_LIMIT_CM:
        return None, f"fitted center drifted {abs(x0 - x0_guess):.1f} cm-1"

    y_fit = lorentzian(x, x0, gamma, A, c)
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    if r_squared < FIT_R2_MIN:
        return None, f"r^2 < {FIT_R2_MIN}"
    area = float(np.pi * A * gamma)

    return (
        {
            "position_cm_inv": x0,
            "height": A,
            "fwhm_cm_inv": fwhm,
            "area_under_peak": area,
            "prominence": float(prominence),
            "r_squared": float(r_squared),
            "baseline_offset": c,
            "n_points_in_window": int(n),
        },
        None,
    )


def fit_all_peaks(
    grid: np.ndarray, intensity: np.ndarray
) -> tuple[list[dict], dict[str, int]]:
    """Find peaks then Lorentzian-fit each. Returns (kept_peaks, reject_counts)."""
    y_for_finder = np.where(np.isfinite(intensity), intensity, 0.0)
    finite_mask = np.isfinite(intensity)
    peaks_idx, props = find_peaks(
        y_for_finder,
        prominence=PEAK_FIND_PROMINENCE,
        distance=PEAK_FIND_DISTANCE,
        width=PEAK_FIND_WIDTH,
    )
    kept: list[dict] = []
    rejects: dict[str, int] = {}
    for k, idx in enumerate(peaks_idx):
        if not finite_mask[idx]:
            rejects["candidate in NaN region"] = (
                rejects.get("candidate in NaN region", 0) + 1
            )
            continue
        peak, reason = fit_one_peak(
            grid,
            intensity,
            int(idx),
            float(props["prominences"][k]),
            float(props["widths"][k]),
        )
        if peak is None:
            rejects[reason or "unknown"] = rejects.get(reason or "unknown", 0) + 1
            continue
        kept.append(peak)
    return kept, rejects


def cluster_by_position(
    peaks: list[dict], tol: float = CLUSTER_TOLERANCE_CM
) -> list[list[dict]]:
    """Single-linkage greedy cluster of peaks by `position_cm_inv`."""
    if not peaks:
        return []
    sorted_peaks = sorted(peaks, key=lambda p: p["position_cm_inv"])
    clusters: list[list[dict]] = [[sorted_peaks[0]]]
    for p in sorted_peaks[1:]:
        last = clusters[-1][-1]
        if p["position_cm_inv"] - last["position_cm_inv"] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


def aggregate_compound_peaks(
    peaks: list[dict],
    n_spectra_total: int,
) -> list[dict]:
    """Cluster + filter by 50% presence; return aggregate peak dicts."""
    clusters = cluster_by_position(peaks, CLUSTER_TOLERANCE_CM)
    threshold = max(1, int(np.ceil(PRESENCE_THRESHOLD * n_spectra_total)))
    aggregates: list[dict] = []
    for cl in clusters:
        spectra_present = {p["spectrum_number"] for p in cl}
        if len(spectra_present) < threshold:
            continue
        aggregates.append(
            {
                "position_cm_inv": float(
                    np.median([p["position_cm_inv"] for p in cl])
                ),
                "height": float(np.median([p["height"] for p in cl])),
                "fwhm_cm_inv": float(
                    np.median([p["fwhm_cm_inv"] for p in cl])
                ),
                "area_under_peak": float(
                    np.median([p["area_under_peak"] for p in cl])
                ),
                "r_squared": float(np.median([p["r_squared"] for p in cl])),
                "n_spectra_present": len(spectra_present),
                "n_spectra_total": int(n_spectra_total),
            }
        )
    return aggregates


def build_cross_compound_table(
    aggregates_by_compound: dict[str, list[dict]],
    compound_totals: dict[str, int],
) -> list[dict]:
    """Cluster aggregate peaks across compounds; return one row per cluster."""
    flat = []
    for compound, aggs in aggregates_by_compound.items():
        for a in aggs:
            flat.append({**a, "compound": compound})
    if not flat:
        return []
    flat.sort(key=lambda p: p["position_cm_inv"])
    clusters: list[list[dict]] = [[flat[0]]]
    for p in flat[1:]:
        last = clusters[-1][-1]
        if p["position_cm_inv"] - last["position_cm_inv"] <= CLUSTER_TOLERANCE_CM:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    rows: list[dict] = []
    for cl in clusters:
        position_bin = float(np.median([p["position_cm_inv"] for p in cl]))
        per_compound: dict[str, dict | None] = {c: None for c in COMPOUND_ORDER}
        for p in cl:
            existing = per_compound[p["compound"]]
            if existing is None or p["height"] > existing["height"]:
                per_compound[p["compound"]] = p
        row = {"position_cm_inv": position_bin}
        for c in COMPOUND_ORDER:
            entry = per_compound[c]
            if entry is None:
                row[c] = None
            else:
                row[c] = (
                    entry["height"],
                    entry["n_spectra_present"],
                    entry["n_spectra_total"],
                )
        rows.append(row)
    return rows


def format_table_for_terminal(rows: list[dict]) -> str:
    """Render the cross-compound table as a human-readable monospace block."""
    col_w = 18
    header = f"{'Position':<10}" + "".join(
        f"{c:<{col_w}}" for c in COMPOUND_ORDER
    )
    out_lines = [header, "-" * len(header)]
    for r in rows:
        line = f"{r['position_cm_inv']:<10.1f}"
        for c in COMPOUND_ORDER:
            entry = r[c]
            if entry is None:
                line += f"{'-':<{col_w}}"
            else:
                height, n_pres, n_tot = entry
                line += f"{f'{height:.3f} ({n_pres}/{n_tot})':<{col_w}}"
        out_lines.append(line)
    return "\n".join(out_lines)


def write_summary_csv(rows: list[dict], path: Path) -> None:
    """Write the cross-compound table as CSV."""
    headers = ["position_cm_inv"] + list(COMPOUND_ORDER)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            cells = [f"{r['position_cm_inv']:.2f}"]
            for c in COMPOUND_ORDER:
                entry = r[c]
                if entry is None:
                    cells.append("")
                else:
                    height, n_pres, n_tot = entry
                    cells.append(f"{height:.3f} ({n_pres}/{n_tot})")
            f.write(",".join(cells) + "\n")


def write_spectra_parquet(rows: dict[str, list], path: Path) -> None:
    """Write reference_spectra.parquet (long form)."""
    schema = pa.schema(
        [
            ("compound", pa.string()),
            ("laser_nm", pa.int32()),
            ("power_mw", pa.int32()),
            ("spectrum_number", pa.string()),
            ("wavenumber_cm_inv", pa.float64()),
            ("intensity_normalized_lab", pa.float64()),
            ("intensity_normalized_convolved", pa.float64()),
            ("usable_for_supercam", pa.bool_()),
        ]
    )
    table = pa.Table.from_pydict(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


def write_catalog_parquet(rows: dict[str, list], path: Path) -> None:
    """Write peak_catalog.parquet (one row per peak, raw + aggregate)."""
    schema = pa.schema(
        [
            ("compound", pa.string()),
            ("laser_nm", pa.int32()),
            ("power_mw", pa.int32()),
            ("spectrum_number", pa.string()),
            ("lab_or_convolved", pa.string()),
            ("position_cm_inv", pa.float64()),
            ("height", pa.float64()),
            ("fwhm_cm_inv", pa.float64()),
            ("area_under_peak", pa.float64()),
            ("prominence", pa.float64()),
            ("r_squared", pa.float64()),
            ("baseline_offset", pa.float64()),
            ("is_aggregate", pa.bool_()),
            ("n_spectra_present", pa.int32()),
            ("n_spectra_total", pa.int32()),
            ("class_label", pa.string()),
            ("fiber_bump_overlap", pa.bool_()),
            ("tolerance_default_cm", pa.float64()),
            ("tolerance_tight_cm", pa.float64()),
        ]
    )
    table = pa.Table.from_pydict(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


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


def write_readme(
    path: Path,
    n_spectra: int,
    n_peaks_fitted: int,
    rejects_total: dict[str, int],
    aggregates_per_compound: dict[str, dict[str, int]],
    warnings_log: list[str],
) -> None:
    if warnings_log:
        head = warnings_log[:50]
        issues_md = "\n".join(f"- {w}" for w in head)
        if len(warnings_log) > 50:
            issues_md += (
                f"\n- ... ({len(warnings_log) - 50} more in `.provenance.json`)"
            )
    else:
        issues_md = "None recorded for the most recent run."

    rejects_md = (
        "\n".join(f"- `{r}`: {n}" for r, n in sorted(rejects_total.items()))
        if rejects_total
        else "- (none)"
    )

    aggregates_md = "\n".join(
        f"- {c}: lab={aggregates_per_compound[c]['lab']}, convolved={aggregates_per_compound[c]['convolved']}"
        for c in COMPOUND_ORDER
        if c in aggregates_per_compound
    )

    text = f"""# Reference library

## What this folder is

SuperCam-resolution reference spectra and a Lorentzian-fit peak catalog for the four iron-based ionic liquid lab compounds. The contents here are the inputs to the matching stage: each row in `peak_catalog.parquet` records a peak's position, height, and width; each row in `reference_spectra.parquet` carries the resampled spectrum at both lab and SuperCam-convolved resolution.

## Source of the inputs

`data/lab_processed/lab_processed.parquet` (16 spectra across 4 compounds and multiple laser/power settings, on the uniform 150-1700 cm-1 / 1 cm-1 grid).

## Build recipe

1. Convolve each lab `intensity_normalized` spectrum with a Gaussian kernel of FWHM 12 cm-1 to match SuperCam's documented Raman resolution (User Guide §2.2). Sigma = FWHM / (2 * sqrt(2 * ln 2)) ≈ {SUPERCAM_SIGMA_CM:.4f} cm-1, and the grid is 1 cm-1 per pixel, so `scipy.ndimage.gaussian_filter1d` is called with `sigma={SUPERCAM_SIGMA_CM:.4f}`. The convolution operates on the contiguous finite block; original NaN positions remain NaN.
2. Re-normalize each convolved spectrum to max = 1 (convolution reduces the peak height of narrow features).
3. Tag each spectrum with `usable_for_supercam = (laser_nm == 532)` since SuperCam's Raman laser is 532 nm; 785 nm spectra remain in the catalog for traceability but are flagged as not directly applicable to the Mars matcher.
4. For each spectrum, run `scipy.signal.find_peaks` on the un-convolved spectrum with `prominence={PEAK_FIND_PROMINENCE}`, `distance={PEAK_FIND_DISTANCE}`, `width={PEAK_FIND_WIDTH}`. Prominence rejects noise; distance prevents double-detecting shoulders on the 1 cm-1 grid; width requires the peak to span at least two grid points, which suppresses any residual cosmic-ray spikes that ALS missed.
5. Fit a Lorentzian profile to each candidate peak in a `+/-{int(FIT_WINDOW_HALF_CM)}` cm-1 window using `scipy.optimize.curve_fit`. Model: `L(x; x0, gamma, A, c) = A * gamma^2 / ((x - x0)^2 + gamma^2) + c`, with HWHM = `gamma`, FWHM = `2 * gamma`, and analytical area above baseline = `pi * A * gamma`. The baseline `c` is bounded to `[{FIT_BASELINE_LOWER}, {FIT_BASELINE_UPPER}]` so `A` reflects peak intensity above the post-ALS zero baseline rather than soaking up wing residuals. Initial guesses come from `find_peaks`. A fit is rejected if `A <= 0`, FWHM > {int(FIT_FWHM_LIMIT_CM)} cm-1, the fitted center drifts more than {int(FIT_X0_DRIFT_LIMIT_CM)} cm-1 from the candidate, or `r^2 < {FIT_R2_MIN}`. The r^2 cutoff filters bad fits (multi-peak windows, fits to noise) without filtering genuine low-intensity peaks; intensity-based detection thresholds are the matcher's concern, not the catalog's.
6. Repeat the fit on the convolved spectrum so the catalog contains peak parameters at both resolutions. The convolved-spectrum row is what the matcher uses; the lab-spectrum row is for cross-resolution comparison.
7. Aggregate per compound. Within a `(compound, lab_or_convolved)` group, single-linkage cluster the per-spectrum peaks by `position_cm_inv` with a {CLUSTER_TOLERANCE_CM:.0f} cm-1 tolerance. A cluster is kept as an aggregate peak if at least {int(PRESENCE_THRESHOLD * 100)}% of the compound's spectra contribute. Aggregate fields are medians: position, FWHM, height, area, r_squared.
8. Cross-compound comparison. Cluster the aggregates again with the same tolerance, build one row per cluster, and write `peak_summary.csv` plus the printed terminal table.

## What was not done

- No class labels in this commit. The `class_label` column is empty; a follow-up commit will populate it by manual annotation.
- No Bayesian peak model.
- No alternate peak shape (Voigt, pseudo-Voigt, asymmetric Lorentzian).
- No asymmetry handling for the fiber-bump residual region.
- No SNR-based weighting in aggregation.

## File conventions

- `reference_spectra.parquet`. Long form, one row per `(compound, laser_nm, power_mw, spectrum_number, wavenumber)`. Carries `intensity_normalized_lab` (un-convolved, copied from `data/lab_processed/`) and `intensity_normalized_convolved` (after Gaussian convolution + re-normalization). `usable_for_supercam` flags the 532 nm subset.
- `peak_catalog.parquet`. One row per `(compound, spectrum_number, peak, lab_or_convolved)` plus aggregate rows. Aggregate rows have `is_aggregate = True`, `spectrum_number = "_aggregate_"`, `laser_nm` and `power_mw` null, and `n_spectra_present` and `n_spectra_total` populated. Per-spectrum (raw) rows have these aggregate fields null. `fiber_bump_overlap` is true for any peak with `200 <= position_cm_inv <= 530`. `tolerance_default_cm` (default 10) and `tolerance_tight_cm` (default 5) carry the matcher's per-peak tolerance windows.
- `peak_summary.csv`. The cross-compound comparison table: `position_cm_inv` plus one column per compound with `{{height}} ({{n_present}}/{{n_total}})`, blank when the compound has no aggregate peak in that cluster.
- `reference_spectra.parquet` and `peak_catalog.parquet` are the canonical artifacts. `peak_summary.csv` is derived for human reading; if it ever disagrees with the catalog, treat the catalog as authoritative.

## Compound abbreviations

| canonical name | formula |
|---|---|
| EMIM-FeCl4 | [EMIM][FeCl4] |
| EMIM-FeBr4 | [EMIM][FeBr4] |
| EMIM2-Fe2Cl7 | [EMIM]2[Fe2Cl7] |
| EMIM2-FeSO4 | [EMIM]2[FeSO4] |

## Provenance

Per-run provenance is written to `.provenance.json` (not tracked). It records the script version, git commit hash, generation timestamp, total spectra processed, total peaks fitted, peak-rejection counts by reason, aggregate-peak counts per compound, and any warnings raised.

This run: {n_spectra} spectra processed, {n_peaks_fitted} peaks fitted across both resolutions.

Aggregate counts:

{aggregates_md}

Rejection breakdown (totals across all spectra and both resolutions):

{rejects_md}

## Known issues

EMIM2-FeSO4 has only 2 lab acquisitions; some aggregate peaks are present in only 1/2 spectra and have correspondingly less stable fit parameters. This is a sample-size limitation, not a pipeline bug.

Per-run warnings:

{issues_md}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    lab_parquet = REPO_ROOT / "data" / "lab_processed" / "lab_processed.parquet"
    out_root = REPO_ROOT / "data" / "reference_library"

    if not lab_parquet.is_file():
        raise FileNotFoundError(f"Missing lab Parquet: {lab_parquet}")

    print(f"Reading {lab_parquet.relative_to(REPO_ROOT)} ...")
    lab_df = pq.read_table(lab_parquet).to_pandas()

    expected_cols = {
        "compound",
        "laser_nm",
        "power_mw",
        "spectrum_number",
        "wavenumber_cm_inv",
        "intensity_normalized",
    }
    missing = expected_cols - set(lab_df.columns)
    if missing:
        raise ValueError(f"Lab Parquet missing columns: {sorted(missing)}")

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    spectra_columns: dict[str, list] = {
        col: []
        for col in [
            "compound",
            "laser_nm",
            "power_mw",
            "spectrum_number",
            "wavenumber_cm_inv",
            "intensity_normalized_lab",
            "intensity_normalized_convolved",
            "usable_for_supercam",
        ]
    }
    catalog_rows: list[dict] = []
    rejects_total: dict[str, int] = {}
    warnings_log: list[str] = []
    n_peaks_fitted = 0

    spectra_per_compound: dict[str, set[str]] = {c: set() for c in COMPOUND_ORDER}
    peaks_for_compound: dict[str, dict[str, list[dict]]] = {
        c: {variant: [] for variant in LAB_OR_CONVOLVED} for c in COMPOUND_ORDER
    }

    group_keys = list(
        lab_df[["compound", "laser_nm", "power_mw", "spectrum_number"]]
        .drop_duplicates()
        .sort_values(by=["compound", "laser_nm", "power_mw", "spectrum_number"])
        .itertuples(index=False, name=None)
    )
    print(f"Processing {len(group_keys)} spectra ...")

    for compound, laser_nm, power_mw, spec_num in group_keys:
        sub = lab_df[
            (lab_df["compound"] == compound)
            & (lab_df["laser_nm"] == laser_nm)
            & (lab_df["power_mw"] == power_mw)
            & (lab_df["spectrum_number"] == spec_num)
        ].sort_values("wavenumber_cm_inv")

        wn = sub["wavenumber_cm_inv"].to_numpy(dtype=float)
        if not np.array_equal(wn, GRID):
            raise ValueError(
                f"Spectrum {compound}/{spec_num} grid does not match lab grid"
            )

        intensity_lab = sub["intensity_normalized"].to_numpy(dtype=float)
        intensity_convolved = gaussian_convolve(
            intensity_lab, sigma_px=SUPERCAM_SIGMA_CM
        )
        # Re-normalize convolved to max = 1.
        peak_conv = np.nanmax(intensity_convolved)
        if np.isfinite(peak_conv) and peak_conv > 0.0:
            intensity_convolved = intensity_convolved / peak_conv

        usable = laser_nm == SUPERCAM_LASER_NM

        n_pts = GRID.size
        spectra_columns["compound"].extend([compound] * n_pts)
        spectra_columns["laser_nm"].extend([int(laser_nm)] * n_pts)
        spectra_columns["power_mw"].extend([int(power_mw)] * n_pts)
        spectra_columns["spectrum_number"].extend([spec_num] * n_pts)
        spectra_columns["wavenumber_cm_inv"].extend(GRID.tolist())
        spectra_columns["intensity_normalized_lab"].extend(intensity_lab.tolist())
        spectra_columns["intensity_normalized_convolved"].extend(
            intensity_convolved.tolist()
        )
        spectra_columns["usable_for_supercam"].extend([bool(usable)] * n_pts)

        spectra_per_compound.setdefault(compound, set()).add(spec_num)

        for variant, intensity in (
            ("lab", intensity_lab),
            ("convolved", intensity_convolved),
        ):
            kept, rejects = fit_all_peaks(GRID, intensity)
            for k, v in rejects.items():
                rejects_total[k] = rejects_total.get(k, 0) + v
            for peak in kept:
                row = {
                    "compound": compound,
                    "laser_nm": int(laser_nm),
                    "power_mw": int(power_mw),
                    "spectrum_number": spec_num,
                    "lab_or_convolved": variant,
                    "is_aggregate": False,
                    "n_spectra_present": None,
                    "n_spectra_total": None,
                    "class_label": "",
                    "fiber_bump_overlap": (
                        FIBER_BUMP_LO
                        <= peak["position_cm_inv"]
                        <= FIBER_BUMP_HI
                    ),
                    "tolerance_default_cm": TOLERANCE_DEFAULT_CM,
                    "tolerance_tight_cm": TOLERANCE_TIGHT_CM,
                    **peak,
                }
                # n_points_in_window is a diagnostic, not in the schema; drop it.
                row.pop("n_points_in_window", None)
                catalog_rows.append(row)
                n_peaks_fitted += 1
                peaks_for_compound.setdefault(
                    compound, {v: [] for v in LAB_OR_CONVOLVED}
                ).setdefault(variant, []).append(
                    {**peak, "spectrum_number": spec_num}
                )

    aggregates_per_compound: dict[str, dict[str, int]] = {}
    aggregates_by_compound_convolved: dict[str, list[dict]] = {}
    compound_totals: dict[str, int] = {
        c: len(spectra_per_compound.get(c, set())) for c in COMPOUND_ORDER
    }

    for compound in COMPOUND_ORDER:
        aggregates_per_compound[compound] = {}
        for variant in LAB_OR_CONVOLVED:
            n_total = compound_totals[compound]
            if n_total == 0:
                aggregates_per_compound[compound][variant] = 0
                continue
            peaks = peaks_for_compound.get(compound, {}).get(variant, [])
            aggs = aggregate_compound_peaks(peaks, n_total)
            aggregates_per_compound[compound][variant] = len(aggs)
            if variant == "convolved":
                aggregates_by_compound_convolved[compound] = aggs
            for a in aggs:
                row = {
                    "compound": compound,
                    "laser_nm": None,
                    "power_mw": None,
                    "spectrum_number": "_aggregate_",
                    "lab_or_convolved": variant,
                    "position_cm_inv": a["position_cm_inv"],
                    "height": a["height"],
                    "fwhm_cm_inv": a["fwhm_cm_inv"],
                    "area_under_peak": a["area_under_peak"],
                    "prominence": None,
                    "r_squared": a["r_squared"],
                    "baseline_offset": None,
                    "is_aggregate": True,
                    "n_spectra_present": int(a["n_spectra_present"]),
                    "n_spectra_total": int(a["n_spectra_total"]),
                    "class_label": "",
                    "fiber_bump_overlap": (
                        FIBER_BUMP_LO
                        <= a["position_cm_inv"]
                        <= FIBER_BUMP_HI
                    ),
                    "tolerance_default_cm": TOLERANCE_DEFAULT_CM,
                    "tolerance_tight_cm": TOLERANCE_TIGHT_CM,
                }
                catalog_rows.append(row)
                if a["r_squared"] < 0.95:
                    warnings_log.append(
                        f"low-R^2 aggregate: {compound} {variant} "
                        f"@ {a['position_cm_inv']:.1f} cm-1 "
                        f"(r2={a['r_squared']:.3f})"
                    )

    # Build the cross-compound table from convolved aggregates.
    cross_rows = build_cross_compound_table(
        aggregates_by_compound_convolved, compound_totals
    )

    print()
    print("Cross-compound aggregate peaks (convolved spectra):")
    print(format_table_for_terminal(cross_rows))

    write_summary_csv(cross_rows, out_root / "peak_summary.csv")

    write_spectra_parquet(spectra_columns, out_root / "reference_spectra.parquet")

    catalog_schema_cols = [
        "compound",
        "laser_nm",
        "power_mw",
        "spectrum_number",
        "lab_or_convolved",
        "position_cm_inv",
        "height",
        "fwhm_cm_inv",
        "area_under_peak",
        "prominence",
        "r_squared",
        "baseline_offset",
        "is_aggregate",
        "n_spectra_present",
        "n_spectra_total",
        "class_label",
        "fiber_bump_overlap",
        "tolerance_default_cm",
        "tolerance_tight_cm",
    ]
    catalog_columns = {col: [] for col in catalog_schema_cols}
    for row in catalog_rows:
        for col in catalog_schema_cols:
            catalog_columns[col].append(row.get(col))
    write_catalog_parquet(
        catalog_columns, out_root / "peak_catalog.parquet"
    )

    write_readme(
        out_root / "README.md",
        n_spectra=len(group_keys),
        n_peaks_fitted=n_peaks_fitted,
        rejects_total=rejects_total,
        aggregates_per_compound=aggregates_per_compound,
        warnings_log=warnings_log,
    )

    n_rejected_total = sum(rejects_total.values())
    provenance = {
        "script": "scripts/04_build_reference_library.py",
        "git_commit_hash": git_head(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_spectra_processed": len(group_keys),
        "n_peaks_fitted": n_peaks_fitted,
        "n_peaks_rejected": n_rejected_total,
        "n_peaks_rejected_by_reason": rejects_total,
        "n_aggregate_peaks_per_compound": aggregates_per_compound,
        "warnings": warnings_log,
    }
    (out_root / ".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(
        f"Spectra processed: {len(group_keys)}; "
        f"peaks fitted: {n_peaks_fitted}; rejected: {n_rejected_total}."
    )
    if rejects_total:
        for reason, n in sorted(rejects_total.items()):
            print(f"  rejected ({reason}): {n}")
    print("Aggregate peaks per compound:")
    for c in COMPOUND_ORDER:
        d = aggregates_per_compound.get(c, {"lab": 0, "convolved": 0})
        print(f"  {c}: lab={d['lab']}, convolved={d['convolved']}")
    if warnings_log:
        print(f"{len(warnings_log)} warning(s) recorded in README and provenance.")


if __name__ == "__main__":
    main()
