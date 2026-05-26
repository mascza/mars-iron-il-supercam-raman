"""Mars Raman ingestion: SuperCam CDR FITS to lab-grid Parquet.

Walks `data/scam_raman_cdrs/`, parses each FITS with the documented PDS schema
(Mars 2020 SuperCam User Guide v5.0), applies the same 5-step recipe used on
lab data, and resamples onto the same uniform 1 cm-1 grid.

References:
  Eilers, P.H.C., Boelens, H.F.M. (2005). Baseline correction with asymmetric
  least squares smoothing.
  Mars 2020 SuperCam Calibration and Data User Guide v5.0; sections 6.1.2.2
  (fiber-bump baseline artifact, 200-530 cm-1) and 6.1.3.1 (Raman wavelength
  axis is in cm-1 despite the column name 'Wavelength').
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.utils.exceptions import AstropyUserWarning

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _processing import (  # noqa: E402
    als_baseline,
    cosmic_ray_remove,
    normalize_to_max,
    resample_to_grid,
    trim_to_range,
)

SCAM_RE = re.compile(
    r"^scam_(\d{4})_(\d{10})_(\d{3})_cr([0-2])_(.{9})_(.{20})_(\d{2})p(\d{2})\.fits$",
    re.IGNORECASE,
)

LASER_HEADER_KEYS = ("LSRWLNTH", "LASER_WL", "EX_WL", "WAVELENG")

GRID = np.arange(150.0, 1701.0, 1.0)
TRIM_LO_CM = 150.0
TRIM_HI_CM = 1700.0
WAVELENGTH_RANGE_OK = (50.0, 8000.0)
NM_RANGE_HINT = (535.0, 853.0)

CR_PRIORITY = {1: 0, 2: 1, 0: 2}


def parse_filename(name: str) -> dict | None:
    """Parse a SCAM CDR filename. Returns parsed fields or None on no match."""
    m = SCAM_RE.match(name)
    if not m:
        return None
    return {
        "sol": int(m.group(1)),
        "sclk_int": int(m.group(2)),
        "sclk_frac": int(m.group(3)),
        "cr_level": int(m.group(4)),
        "seqid": m.group(5).lower(),
        "target": m.group(6).rstrip("_"),
        "point": int(m.group(7)),
        "version": int(m.group(8)),
    }


def discover_and_dedup(
    raw_dir: Path,
) -> tuple[list[tuple[Path, dict]], dict[str, int], list[str], int, int]:
    """Walk raw_dir, parse filenames, group and pick CR1 > CR2 > CR0.

    Returns (chosen_list, cr_distribution, warnings, n_seen, n_skipped).
    """
    warnings_log: list[str] = []
    files_by_key: dict[tuple, list[tuple[Path, dict]]] = {}
    n_seen = 0
    n_skipped = 0

    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file():
            continue
        if not p.name.lower().endswith(".fits"):
            continue
        n_seen += 1
        meta = parse_filename(p.name)
        if meta is None:
            warnings_log.append(
                f"unparseable filename: {p.relative_to(REPO_ROOT)}"
            )
            n_skipped += 1
            continue
        key = (
            meta["sol"],
            meta["sclk_int"],
            meta["sclk_frac"],
            meta["seqid"],
            meta["point"],
        )
        files_by_key.setdefault(key, []).append((p, meta))

    chosen: list[tuple[Path, dict]] = []
    cr_dist: dict[str, int] = {"CR0": 0, "CR1": 0, "CR2": 0}
    for key in sorted(files_by_key.keys()):
        candidates = files_by_key[key]
        candidates.sort(key=lambda c: CR_PRIORITY.get(c[1]["cr_level"], 99))
        path, meta = candidates[0]
        cr_dist[f"CR{meta['cr_level']}"] += 1
        chosen.append((path, meta))

    return chosen, cr_dist, warnings_log, n_seen, n_skipped


def smoke_test(fits_path: Path) -> tuple[bool, list[str]]:
    """Exercise structural assumptions on the first FITS file.

    Returns (ok, messages). Each message is one line of human-readable output.
    """
    msgs: list[str] = [f"target: {fits_path.name}"]
    ok = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            hdul = fits.open(fits_path)
        except Exception as exc:
            return False, msgs + [f"FAIL: open raised {type(exc).__name__}: {exc}"]
        unexpected = [
            w
            for w in caught
            if not issubclass(w.category, (VerifyWarning, AstropyUserWarning))
        ]
        if unexpected:
            ok = False
            msgs.append(
                f"FAIL: open emitted {len(unexpected)} unexpected warning(s): "
                + "; ".join(
                    f"{w.category.__name__}: {w.message}" for w in unexpected
                )
            )

    try:
        if len(hdul) < 9:
            ok = False
            msgs.append(
                f"FAIL: HDU list has {len(hdul)} HDUs, expected at least 9"
            )
        else:
            msgs.append(f"PASS: HDU count = {len(hdul)}")

        d7 = hdul[7].data
        if d7 is None or d7.columns is None:
            ok = False
            msgs.append("FAIL: HDU 7 has no table data")
        else:
            cols7 = set(d7.columns.names)
            missing = {"Mean", "Median", "StDev"} - cols7
            if missing:
                ok = False
                msgs.append(f"FAIL: HDU 7 missing columns: {sorted(missing)}")
            else:
                msgs.append(
                    f"PASS: HDU 7 has Mean, Median, StDev ({len(d7)} rows)"
                )

        d8 = hdul[8].data
        if (
            d8 is None
            or d8.columns is None
            or "Wavelength" not in d8.columns.names
        ):
            ok = False
            msgs.append("FAIL: HDU 8 missing 'Wavelength' column")
        else:
            wave = np.asarray(d8["Wavelength"], dtype=float)
            if wave.size < 2:
                ok = False
                msgs.append("FAIL: HDU 8 Wavelength has fewer than 2 points")
            elif not np.all(np.isfinite(wave)):
                ok = False
                msgs.append("FAIL: HDU 8 Wavelength contains NaN or inf")
            else:
                sorted_wave = np.sort(wave)
                if not bool((np.diff(sorted_wave) > 0).all()):
                    ok = False
                    msgs.append(
                        "FAIL: HDU 8 Wavelength has duplicate values; "
                        "cannot establish monotonic axis after sort"
                    )
                else:
                    wmin = float(sorted_wave[0])
                    wmax = float(sorted_wave[-1])
                    lo, hi = WAVELENGTH_RANGE_OK
                    nm_lo, nm_hi = NM_RANGE_HINT
                    if nm_lo <= wmin <= nm_hi and nm_lo <= wmax <= nm_hi:
                        ok = False
                        msgs.append(
                            f"FAIL: Wavelength looks like nm "
                            f"({wmin:.1f}-{wmax:.1f}); Raman CDR is supposed "
                            "to be cm-1 per User Guide §6.1.3.1"
                        )
                    elif wmin < lo or wmax > hi:
                        ok = False
                        msgs.append(
                            f"FAIL: Wavelength range "
                            f"{wmin:.1f}-{wmax:.1f} outside expected "
                            f"[{lo}, {hi}] cm-1"
                        )
                    else:
                        msgs.append(
                            f"PASS: Wavelength sortable, in cm-1 "
                            f"({wmin:.1f}-{wmax:.1f}, n={wave.size})"
                        )

        h0 = hdul[0].header
        if not h0:
            ok = False
            msgs.append("FAIL: primary header is empty")
        else:
            msgs.append(f"PASS: primary header has {len(h0)} cards")
    finally:
        hdul.close()

    return ok, msgs


def get_first_present(header, keys):
    """Return the first non-None header value for any key in keys."""
    for k in keys:
        v = header.get(k)
        if v is not None:
            return v
    return None


def process_observation(fits_path: Path, grid: np.ndarray) -> dict:
    """Apply the 5-step recipe to Mean and Median; return arrays + scalars."""
    with fits.open(fits_path) as hdul:
        d7 = hdul[7].data
        d8 = hdul[8].data
        h0 = hdul[0].header

        mean = np.asarray(d7["Mean"], dtype=float)
        median = np.asarray(d7["Median"], dtype=float)
        wave = np.asarray(d8["Wavelength"], dtype=float)

        # Detect inversions (descending steps) BEFORE sorting. The user guide
        # §6.1.2.2 documents 1-2 such inversions per file at the green/orange/
        # red detector seams; many or wildly varying indices indicate a real
        # anomaly rather than the standard stitch.
        inversion_indices = np.where(np.diff(wave) < 0.0)[0].tolist()

        order = np.argsort(wave, kind="stable")
        wave = wave[order]
        mean = mean[order]
        median = median[order]

        native_min = float(wave[0])
        native_max = float(wave[-1])
        oor_low = native_min > grid[0]
        oor_high = native_max < grid[-1]

        m_clean = cosmic_ray_remove(mean)
        m_corr = m_clean - als_baseline(m_clean)
        x_t, y_t = trim_to_range(wave, m_corr, TRIM_LO_CM, TRIM_HI_CM)
        m_corr_grid = (
            resample_to_grid(x_t, y_t, grid)
            if x_t.size
            else np.full(grid.shape, np.nan)
        )
        m_norm = normalize_to_max(m_corr_grid)

        med_clean = cosmic_ray_remove(median)
        med_corr = med_clean - als_baseline(med_clean)
        x_t, y_t = trim_to_range(wave, med_corr, TRIM_LO_CM, TRIM_HI_CM)
        med_corr_grid = (
            resample_to_grid(x_t, y_t, grid)
            if x_t.size
            else np.full(grid.shape, np.nan)
        )
        med_norm = normalize_to_max(med_corr_grid)

        laser_raw = get_first_present(h0, LASER_HEADER_KEYS)

    laser_wl: float | None
    if laser_raw is None:
        laser_wl = None
    else:
        try:
            laser_wl = float(laser_raw)
        except (TypeError, ValueError):
            laser_wl = None

    return {
        "intensity_baseline_corrected_mean": m_corr_grid,
        "intensity_normalized_mean": m_norm,
        "intensity_baseline_corrected_median": med_corr_grid,
        "intensity_normalized_median": med_norm,
        "out_of_range_low": oor_low,
        "out_of_range_high": oor_high,
        "laser_wavelength_nm": laser_wl,
        "native_min_cm_inv": native_min,
        "native_max_cm_inv": native_max,
        "inversion_count": len(inversion_indices),
        "inversion_indices": inversion_indices,
    }


def safe_target(s: str) -> str:
    """Filename-safe target name. Returns 'unknown' on empty input."""
    if not s:
        return "unknown"
    s = s.rstrip("_").strip()
    if not s:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


def mrow_get(mrow: dict | None, key: str, conv=None):
    """Pull a value from a masterlist row dict, returning None when missing/NaN."""
    if mrow is None:
        return None
    v = mrow.get(key)
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if conv is None:
        return v if isinstance(v, str) else v
    try:
        return conv(v)
    except (TypeError, ValueError):
        return None


def write_csv(
    path: Path,
    grid: np.ndarray,
    cols: dict[str, np.ndarray],
) -> None:
    """Write a Mars CSV with `wavenumber_cm_inv` + the four intensity columns."""
    headers = ["wavenumber_cm_inv"] + list(cols.keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for i in range(grid.size):
            row = [f"{float(grid[i]):.2f}"]
            for k in cols:
                v = float(cols[k][i])
                row.append("nan" if not np.isfinite(v) else f"{v:.6f}")
            f.write(",".join(row) + "\n")


def write_parquet(columns: dict[str, list], path: Path) -> None:
    """Materialize the long-form Mars Parquet (zstd-compressed)."""
    schema = pa.schema(
        [
            ("sol", pa.int32()),
            ("sclk_int", pa.int64()),
            ("sclk_frac", pa.int32()),
            ("seqid", pa.string()),
            ("point", pa.int32()),
            ("cr_level", pa.int32()),
            ("target_name", pa.string()),
            ("tdb_name", pa.string()),
            ("target_classification", pa.string()),
            ("n_accumulations", pa.int32()),
            ("n_active_spectra", pa.int32()),
            ("integration_time_us", pa.float64()),
            ("lmst", pa.string()),
            ("laser_wavelength_nm", pa.float64()),
            ("source_filename", pa.string()),
            ("source_relpath", pa.string()),
            ("wavenumber_cm_inv", pa.float64()),
            ("intensity_baseline_corrected_mean", pa.float64()),
            ("intensity_normalized_mean", pa.float64()),
            ("intensity_baseline_corrected_median", pa.float64()),
            ("intensity_normalized_median", pa.float64()),
            ("out_of_range_low", pa.bool_()),
            ("out_of_range_high", pa.bool_()),
        ]
    )
    table = pa.Table.from_pydict(columns, schema=schema)
    pq.write_table(table, path, compression="zstd")


def git_head() -> str:
    """Current git HEAD or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_readme(path: Path, run_summary: dict) -> None:
    n_obs = run_summary["n_observations_processed"]
    n_files = run_summary["n_files_seen"]
    cr_dist = run_summary["cr_level_distribution"]
    n_disk_orphans = run_summary["n_disk_orphans"]
    n_mlist_orphans = run_summary["n_masterlist_orphans"]
    warnings_list = run_summary["warnings"]

    if warnings_list:
        head = warnings_list[:50]
        issues_md = "\n".join(f"- {w}" for w in head)
        if len(warnings_list) > 50:
            issues_md += (
                f"\n- ... ({len(warnings_list) - 50} more in `.provenance.json`)"
            )
    else:
        issues_md = "None recorded for the most recent run."

    text = f"""# Mars-processed Raman spectra

## What this folder is

Processed Mars Raman spectra from the SuperCam corpus, derived from CDR FITS files in `data/scam_raman_cdrs/`. The contents here are the Mars-side counterpart to `data/lab_processed/`: same wavenumber grid, same processing recipe, ready for direct matching against the lab reference library.

## Source of the raw data

Raw FITS files live in `data/scam_raman_cdrs/sol_NNNNN/scam_*.fits`; these are gitignored and archived externally. Observation metadata comes from `data/supercam/m2020_scam_masterlist.csv`. The FITS schema follows the Mars 2020 SuperCam User Guide v5.0: HDU 7 is Statistics (`Mean`, `Median`, `StDev`), HDU 8 is Wavelength (already in cm-1 per §6.1.3.1).

## Processing recipe

1. Cosmic ray removal. Discrete second derivative; flag points whose deviation from the median exceeds `8 * MAD`; replace with the median of a 5-point neighborhood; iterate up to 3 times.
2. ALS baseline correction. Asymmetric least squares with `lam = 1e5`, `p = 0.001`, `niter = 10`. Reference: Eilers and Boelens (2005). The fiber-bump baseline artifact in 200-530 cm-1 (User Guide §6.1.2.2) is absorbed by the ALS fit; no separate model is fitted.
3. Trim to `150 <= wavenumber <= 1700 cm-1`.
4. Resample to the lab grid: `np.arange(150, 1701, 1.0)`, 1551 points at 1 cm-1 spacing. Linear interpolation. Positions outside the native range are NaN; no extrapolation.
5. Normalize by the NaN-ignoring max of the resampled, baseline-corrected spectrum.

The recipe runs separately on the Mean and Median statistics arrays from HDU 7; both sets of intermediates and normalized values appear in the Parquet.

## What was not done

- No fiber-bump model fitting; the artifact is absorbed into the ALS baseline.
- No wavelength recalibration. The User Guide notes calibration uncertainty up to ~2.2 cm-1 at 1330 cm-1 for high-SNR spectra and several cm-1 for low-SNR.
- No CR-level merging within an observation; one representative CR file is chosen per observation.
- No spectrum-quality filtering.
- No saturation handling beyond what the SuperCam CDR pipeline already applied.

## Discovery and deduplication

Filenames are parsed with a case-insensitive regex matching `scam_NNNN_NNNNNNNNNN_NNN_crN_<seqid 9>_<target 20>_NNpNN.fits`. Files are grouped by the unique observation key `(sol, sclk_int, sclk_frac, seqid, point)`. Within a group, the representative file is chosen by `CR1 > CR2 > CR0`. Total source files seen: {n_files}. Unique observations: {n_obs}. CR-level breakdown: CR0={cr_dist["CR0"]}, CR1={cr_dist["CR1"]}, CR2={cr_dist["CR2"]}.

## File conventions

CSVs are written to `spectra/` with names of the form:

    sol{{sol:04d}}_sclk{{sclk_int:010d}}_{{seqid}}_pt{{point:02d}}_{{target}}.csv

The masterlist join is on the tuple `(sol, sclk_int, seqid, point_number)`, not on the filename string. The masterlist's `cdr_fname` mixes upper and lower case and ships P02, P03, and P01 versions where disk files use lowercase and may carry different version suffixes; tuple keying sidesteps the case and version mismatch.

CSV columns: `wavenumber_cm_inv, intensity_baseline_corrected_mean, intensity_normalized_mean, intensity_baseline_corrected_median, intensity_normalized_median`. Two decimals on wavenumber, six on intensities. UTF-8.

The Parquet `mars_processed.parquet` (one row per (observation, wavenumber)) has the schema:

| column | dtype | source |
|---|---|---|
| `sol` | int32 | filename |
| `sclk_int` | int64 | filename |
| `sclk_frac` | int32 | filename |
| `seqid` | string | filename |
| `point` | int32 | filename |
| `cr_level` | int32 | filename |
| `target_name` | string | masterlist `targetname`, fallback to filename target |
| `tdb_name` | string | masterlist `tdb_name`, nullable |
| `target_classification` | string | masterlist `Target_Type`, nullable |
| `n_accumulations` | int32 | masterlist `actives_ncoadds`, nullable |
| `n_active_spectra` | int32 | masterlist `actives_nspectra`, nullable |
| `integration_time_us` | float64 | masterlist `t_integ_real`, nullable |
| `lmst` | string | masterlist, nullable |
| `laser_wavelength_nm` | float64 | primary header, nullable |
| `source_filename` | string | basename only |
| `source_relpath` | string | path relative to repo root |
| `wavenumber_cm_inv` | float64 | uniform 150-1700, 1 cm-1 spacing |
| `intensity_baseline_corrected_mean` | float64 | post recipe step 4, from HDU 7 Mean |
| `intensity_normalized_mean` | float64 | post recipe step 5, from HDU 7 Mean |
| `intensity_baseline_corrected_median` | float64 | post recipe step 4, from HDU 7 Median |
| `intensity_normalized_median` | float64 | post recipe step 5, from HDU 7 Median |
| `out_of_range_low` | bool | true if native range starts above 150 |
| `out_of_range_high` | bool | true if native range ends below 1700 |

The Parquet is canonical. CSVs are derived from the same in-memory arrays at write time. If they ever disagree, treat the Parquet as authoritative.

## Provenance

Provenance for each script run is written to `.provenance.json` (not tracked in git). It records the script version, git commit hash, generation timestamp, file and observation counts, CR-level distribution, masterlist join orphan counts, and any warnings raised.

## Known issues

Disk files without a masterlist entry: {n_disk_orphans}. Masterlist Raman entries with no file on disk: {n_mlist_orphans}.

{issues_md}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    raw_dir = REPO_ROOT / "data" / "scam_raman_cdrs"
    masterlist_path = REPO_ROOT / "data" / "supercam" / "m2020_scam_masterlist.csv"
    out_root = REPO_ROOT / "data" / "mars_processed"

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw FITS dir: {raw_dir}")
    if not masterlist_path.is_file():
        raise FileNotFoundError(f"Missing masterlist: {masterlist_path}")

    print(f"Scanning {raw_dir.relative_to(REPO_ROOT)} ...")
    chosen, cr_dist, dedup_warnings, n_seen, n_skipped = discover_and_dedup(raw_dir)

    if not chosen:
        print("ERROR: no parseable FITS files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Total source files seen: {n_seen}")
    print(f"Unique observations: {len(chosen)}")
    print(f"  CR1: {cr_dist['CR1']}")
    print(f"  CR2: {cr_dist['CR2']}")
    print(f"  CR0: {cr_dist['CR0']}")
    if n_seen:
        print(f"Deduplication factor: {n_seen / len(chosen):.2f}")
    if n_skipped:
        print(f"Files skipped (unparseable): {n_skipped}")

    if len(chosen) < 100 or len(chosen) > 2000:
        print(
            f"ERROR: unique observation count {len(chosen)} outside expected "
            "[100, 2000] window; aborting before writing any output.",
            file=sys.stderr,
        )
        sys.exit(1)

    print()
    smoke_ok, smoke_msgs = smoke_test(chosen[0][0])
    for m in smoke_msgs:
        print("  " + m)
    if not smoke_ok:
        print("Smoke test FAILED. No output written.", file=sys.stderr)
        sys.exit(1)
    print("Smoke test PASSED.")

    lab_parquet = REPO_ROOT / "data" / "lab_processed" / "lab_processed.parquet"
    if lab_parquet.exists():
        lab_table = pq.read_table(lab_parquet, columns=["wavenumber_cm_inv"])
        lab_grid_first = lab_table.column("wavenumber_cm_inv").to_numpy()[: GRID.size]
        if not np.array_equal(lab_grid_first, GRID):
            print(
                "ERROR: lab Parquet wavenumber grid does not equal "
                "np.arange(150, 1701, 1.0); refusing to process.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("Lab grid identity check: PASS")
    else:
        print("Note: lab Parquet not found; skipping grid identity check.")

    print()
    mlist = pd.read_csv(masterlist_path, encoding="utf-8-sig", low_memory=False)
    raman_mlist = mlist[mlist["type"] == "RAMAN"].copy()
    print(f"Masterlist Raman entries: {len(raman_mlist)}")
    if len(raman_mlist) != 1436:
        print(
            f"  WARNING: expected 1436 Raman entries, got {len(raman_mlist)}"
        )

    mlist_lookup: dict[tuple, dict] = {}
    for _, row in raman_mlist.iterrows():
        cdr = row.get("cdr_fname")
        if not isinstance(cdr, str):
            continue
        meta = parse_filename(cdr)
        if meta is None:
            continue
        key = (meta["sol"], meta["sclk_int"], meta["seqid"], meta["point"])
        mlist_lookup.setdefault(key, row.to_dict())

    matched_for_print: list[tuple[str, tuple]] = []
    n_matches = 0
    for path, meta in chosen:
        key = (meta["sol"], meta["sclk_int"], meta["seqid"], meta["point"])
        if key in mlist_lookup:
            n_matches += 1
            if len(matched_for_print) < 3:
                matched_for_print.append((path.name, key))

    if n_matches == 0:
        print("ERROR: 0 matches between disk and masterlist.")
        if chosen:
            d = chosen[0][1]
            print(
                f"  first disk tuple: "
                f"({d['sol']}, {d['sclk_int']}, {d['seqid']!r}, {d['point']})"
            )
        if mlist_lookup:
            print(f"  first masterlist tuple: {next(iter(mlist_lookup.keys()))}")
        sys.exit(1)

    print("First 3 matched observations:")
    for name, key in matched_for_print:
        print(f"  {name} -> {key}")
    pct = 100.0 * n_matches / len(chosen)
    print(
        f"Disk observations with a masterlist hit: "
        f"{n_matches}/{len(chosen)} ({pct:.1f}%)"
    )
    if pct < 50.0:
        print(
            f"ERROR: masterlist join rate {pct:.1f}% below 50% threshold; "
            "aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    if out_root.exists():
        shutil.rmtree(out_root)
    spectra_dir = out_root / "spectra"
    spectra_dir.mkdir(parents=True)

    print()
    print(f"Processing {len(chosen)} observations ...")

    columns: dict[str, list] = {
        col: []
        for col in [
            "sol",
            "sclk_int",
            "sclk_frac",
            "seqid",
            "point",
            "cr_level",
            "target_name",
            "tdb_name",
            "target_classification",
            "n_accumulations",
            "n_active_spectra",
            "integration_time_us",
            "lmst",
            "laser_wavelength_nm",
            "source_filename",
            "source_relpath",
            "wavenumber_cm_inv",
            "intensity_baseline_corrected_mean",
            "intensity_normalized_mean",
            "intensity_baseline_corrected_median",
            "intensity_normalized_median",
            "out_of_range_low",
            "out_of_range_high",
        ]
    }
    seen_csv_names: set[str] = set()
    warnings_log: list[str] = list(dedup_warnings)
    inversion_count_distribution: dict[int, int] = {}
    n_processed = 0
    n_with_warning = 0
    n_pts = GRID.size
    start = time.time()

    for idx, (path, meta) in enumerate(chosen, start=1):
        try:
            result = process_observation(path, GRID)
        except Exception as exc:
            warnings_log.append(
                f"FAIL processing {path.name}: {type(exc).__name__}: {exc}"
            )
            n_with_warning += 1
            continue

        key = (meta["sol"], meta["sclk_int"], meta["seqid"], meta["point"])
        mrow = mlist_lookup.get(key)
        had_warning = False

        if mrow is None:
            warnings_log.append(f"no masterlist entry for {path.name}")
            had_warning = True

        target_full = mrow_get(mrow, "targetname")
        if not isinstance(target_full, str) or not target_full.strip():
            target_full = meta.get("target") or "unknown"
            if target_full == "unknown":
                warnings_log.append(
                    f"no target name for {path.name}; using 'unknown'"
                )
                had_warning = True

        target_name_canonical = (
            target_full.strip() if isinstance(target_full, str) else "unknown"
        )
        target_safe = safe_target(target_name_canonical)

        if result["out_of_range_low"] or result["out_of_range_high"]:
            warnings_log.append(
                f"{path.name}: native range "
                f"{result['native_min_cm_inv']:.1f}-"
                f"{result['native_max_cm_inv']:.1f} cm-1 does not fully cover "
                "[150, 1700]; some grid points NaN"
            )
            had_warning = True

        inv_count = result["inversion_count"]
        inversion_count_distribution[inv_count] = (
            inversion_count_distribution.get(inv_count, 0) + 1
        )
        if inv_count > 0:
            inv_idx = result["inversion_indices"]
            if len(inv_idx) > 10:
                idx_repr = f"{inv_idx[:10]} ... ({len(inv_idx)} total)"
            else:
                idx_repr = str(inv_idx)
            warnings_log.append(
                f"{path.name}: {inv_count} wavelength inversion(s) at "
                f"indices {idx_repr}"
            )
            had_warning = True

        csv_name = (
            f"sol{meta['sol']:04d}_sclk{meta['sclk_int']:010d}_"
            f"{meta['seqid']}_pt{meta['point']:02d}_{target_safe}.csv"
        )
        if csv_name in seen_csv_names:
            raise RuntimeError(
                f"CSV filename collision: {csv_name} (source {path.name})"
            )
        seen_csv_names.add(csv_name)
        write_csv(
            spectra_dir / csv_name,
            GRID,
            {
                "intensity_baseline_corrected_mean": result[
                    "intensity_baseline_corrected_mean"
                ],
                "intensity_normalized_mean": result["intensity_normalized_mean"],
                "intensity_baseline_corrected_median": result[
                    "intensity_baseline_corrected_median"
                ],
                "intensity_normalized_median": result[
                    "intensity_normalized_median"
                ],
            },
        )

        tdb = mrow_get(mrow, "tdb_name")
        target_class = mrow_get(mrow, "Target_Type")
        nacc = mrow_get(mrow, "actives_ncoadds", lambda v: int(float(v)))
        nspec = mrow_get(mrow, "actives_nspectra", lambda v: int(float(v)))
        intt = mrow_get(mrow, "t_integ_real", lambda v: float(v))
        lmst = mrow_get(mrow, "lmst")

        columns["sol"].extend([meta["sol"]] * n_pts)
        columns["sclk_int"].extend([meta["sclk_int"]] * n_pts)
        columns["sclk_frac"].extend([meta["sclk_frac"]] * n_pts)
        columns["seqid"].extend([meta["seqid"]] * n_pts)
        columns["point"].extend([meta["point"]] * n_pts)
        columns["cr_level"].extend([meta["cr_level"]] * n_pts)
        columns["target_name"].extend([target_name_canonical] * n_pts)
        columns["tdb_name"].extend(
            [tdb if isinstance(tdb, str) else None] * n_pts
        )
        columns["target_classification"].extend(
            [target_class if isinstance(target_class, str) else None] * n_pts
        )
        columns["n_accumulations"].extend([nacc] * n_pts)
        columns["n_active_spectra"].extend([nspec] * n_pts)
        columns["integration_time_us"].extend([intt] * n_pts)
        columns["lmst"].extend(
            [lmst if isinstance(lmst, str) else None] * n_pts
        )
        columns["laser_wavelength_nm"].extend(
            [result["laser_wavelength_nm"]] * n_pts
        )
        columns["source_filename"].extend([path.name] * n_pts)
        columns["source_relpath"].extend(
            [str(path.relative_to(REPO_ROOT))] * n_pts
        )
        columns["wavenumber_cm_inv"].extend(GRID.tolist())
        columns["intensity_baseline_corrected_mean"].extend(
            result["intensity_baseline_corrected_mean"].tolist()
        )
        columns["intensity_normalized_mean"].extend(
            result["intensity_normalized_mean"].tolist()
        )
        columns["intensity_baseline_corrected_median"].extend(
            result["intensity_baseline_corrected_median"].tolist()
        )
        columns["intensity_normalized_median"].extend(
            result["intensity_normalized_median"].tolist()
        )
        columns["out_of_range_low"].extend([result["out_of_range_low"]] * n_pts)
        columns["out_of_range_high"].extend(
            [result["out_of_range_high"]] * n_pts
        )

        n_processed += 1
        if had_warning:
            n_with_warning += 1

        if idx % 100 == 0:
            elapsed = time.time() - start
            print(
                f"  ... {idx}/{len(chosen)} processed "
                f"({elapsed:.1f}s elapsed)"
            )

    disk_keys = {
        (meta["sol"], meta["sclk_int"], meta["seqid"], meta["point"])
        for _, meta in chosen
    }
    n_disk_orphans = sum(1 for k in disk_keys if k not in mlist_lookup)
    n_mlist_orphans = sum(1 for k in mlist_lookup if k not in disk_keys)

    parquet_path = out_root / "mars_processed.parquet"
    write_parquet(columns, parquet_path)

    run_summary = {
        "n_observations_processed": n_processed,
        "n_files_seen": n_seen,
        "cr_level_distribution": cr_dist,
        "n_masterlist_orphans": n_mlist_orphans,
        "n_disk_orphans": n_disk_orphans,
        "warnings": warnings_log,
    }
    write_readme(out_root / "README.md", run_summary)

    provenance = {
        "script": "scripts/03_ingest_supercam.py",
        "git_commit_hash": git_head(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_observations_processed": n_processed,
        "n_files_seen": n_seen,
        "n_files_skipped": n_skipped,
        "cr_level_distribution": cr_dist,
        "n_masterlist_orphans": n_mlist_orphans,
        "n_disk_orphans": n_disk_orphans,
        "inversion_count_distribution": {
            str(k): v
            for k, v in sorted(inversion_count_distribution.items())
        },
        "warnings": warnings_log,
    }
    (out_root / ".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    parquet_size = parquet_path.stat().st_size
    elapsed = time.time() - start
    print()
    print(
        f"Wrote {n_processed} observations ({len(seen_csv_names)} CSVs) "
        f"and mars_processed.parquet "
        f"({parquet_size / (1024 * 1024):.1f} MiB) in {elapsed:.1f}s."
    )
    print(
        f"Disk orphans: {n_disk_orphans}; "
        f"masterlist orphans: {n_mlist_orphans}."
    )
    print(
        f"Observations with at least one warning: {n_with_warning}; "
        f"total warning entries: {len(warnings_log)}."
    )


if __name__ == "__main__":
    main()
