"""Apply the lab-spectrum processing recipe to every raw WITec acquisition.

Walks `data/raw_lab/`, runs the 5-step recipe (shared with the Mars side via
`_processing.py`):

  1. Cosmic ray removal: median-of-5 replacement at points where the discrete
     second derivative deviates from its median by more than 8 * MAD,
     iterated up to 3 times. Median-of-N derivative outlier replacement is a
     standard cosmic-ray method (see e.g. Whitaker and Hayes 2018,
     Chemometr. Intell. Lab. Syst. 179: 82).
  2. ALS baseline correction: asymmetric least squares with lam=1e5,
     p=0.001, niter=10. Reference: Eilers, P.H.C., Boelens, H.F.M. (2005),
     "Baseline correction with asymmetric least squares smoothing".
  3. Trim to 150 <= wavenumber <= 1700 cm-1.
  4. Resample to a uniform 1 cm-1 grid (`np.arange(150, 1701, 1.0)`, 1551
     points), linear interpolation, NaN outside the native range.
  5. Normalize by dividing the resampled spectrum by its NaN-ignoring max.

Writes per-acquisition CSVs and a single combined Parquet to
`data/lab_processed/`. Wipes that directory before regenerating.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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

# 01_inspect_witec.py cannot be imported by name (Python identifiers cannot
# start with a digit), so load it from its file path.
_spec = importlib.util.spec_from_file_location(
    "inspect_witec",
    SCRIPTS_DIR / "01_inspect_witec.py",
)
assert _spec is not None and _spec.loader is not None
_inspect_witec = importlib.util.module_from_spec(_spec)
sys.modules["inspect_witec"] = _inspect_witec
_spec.loader.exec_module(_inspect_witec)
Acquisition = _inspect_witec.Acquisition
load_acquisition = _inspect_witec.load_acquisition
lookup = _inspect_witec.lookup


COMPOUND_MAP: dict[str, str] = {
    "emim_fecl4": "EMIM-FeCl4",
    "emim_febr4": "EMIM-FeBr4",
    "emim_fe2cl7": "EMIM2-Fe2Cl7",
    "emim_feso4": "EMIM2-FeSO4",
}

NM_RE = re.compile(r"(\d+)\s*nm", re.IGNORECASE)
MW_RE = re.compile(r"(\d+)\s*mw", re.IGNORECASE)

TRIM_LO_CM = 150.0
TRIM_HI_CM = 1700.0
LAB_GRID = np.arange(TRIM_LO_CM, TRIM_HI_CM + 1.0, 1.0)


def discover_acquisition_folders(raw_lab: Path) -> list[Path]:
    """Find every folder under `raw_lab` that holds WITec acquisition files."""
    folders: set[Path] = set()
    for path in raw_lab.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if not name.endswith(".txt"):
            continue
        if "Export File (X-Axis)" in name or "Spec.Data" in name:
            folders.add(path.parent)
    folders.discard(raw_lab)
    return sorted(folders)


def extract_laser_power_from_name(name: str) -> tuple[int | None, int | None]:
    """Pull integer nm/mW from a folder name. Order-agnostic, case-insensitive."""
    nm_match = NM_RE.search(name)
    mw_match = MW_RE.search(name)
    nm = int(nm_match.group(1)) if nm_match else None
    mw = int(mw_match.group(1)) if mw_match else None
    return nm, mw


def process_acquisition(
    acq: Acquisition,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Run the 5-step recipe.

    Returns (y_baseline_corrected_on_grid, y_normalized_on_grid,
    out_of_range_low, out_of_range_high). Both intensity arrays are aligned
    to `LAB_GRID`.
    """
    y_clean = cosmic_ray_remove(acq.y)
    baseline = als_baseline(y_clean)
    y_corr = y_clean - baseline
    x_trim, y_trim = trim_to_range(acq.x, y_corr, TRIM_LO_CM, TRIM_HI_CM)
    if x_trim.size == 0:
        raise ValueError(
            f"Trimmed range [{TRIM_LO_CM}, {TRIM_HI_CM}] cm-1 is empty for "
            f"{acq.folder.name} / spec {acq.spectrum_number}"
        )
    oor_low = float(x_trim[0]) > LAB_GRID[0]
    oor_high = float(x_trim[-1]) < LAB_GRID[-1]
    y_corr_grid = resample_to_grid(x_trim, y_trim, LAB_GRID)
    y_norm_grid = normalize_to_max(y_corr_grid)
    if not np.any(np.isfinite(y_norm_grid)):
        raise ValueError(
            f"No finite values after resampling {acq.folder.name} / spec "
            f"{acq.spectrum_number}; native range may not overlap grid"
        )
    return y_corr_grid, y_norm_grid, oor_low, oor_high


def write_csv(
    path: Path,
    x: np.ndarray,
    y_corr: np.ndarray,
    y_norm: np.ndarray,
) -> None:
    """Write a 3-column CSV: wavenumber, baseline-corrected, normalized."""
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "wavenumber_cm_inv,intensity_baseline_corrected,intensity_normalized\n"
        )
        for xi, yi, ni in zip(x.tolist(), y_corr.tolist(), y_norm.tolist()):
            f.write(f"{xi:.2f},{yi:.6f},{ni:.6f}\n")


def write_parquet(columns: dict[str, list], path: Path) -> None:
    """Materialize the long-form spectra table as a zstd-compressed Parquet."""
    schema = pa.schema(
        [
            ("compound", pa.string()),
            ("laser_nm", pa.int32()),
            ("power_mw", pa.int32()),
            ("spectrum_number", pa.string()),
            ("source_folder", pa.string()),
            ("sample_name", pa.string()),
            ("sample_inferred", pa.bool_()),
            ("wavenumber_cm_inv", pa.float64()),
            ("intensity_baseline_corrected", pa.float64()),
            ("intensity_normalized", pa.float64()),
            ("out_of_range_low", pa.bool_()),
            ("out_of_range_high", pa.bool_()),
        ]
    )
    table = pa.Table.from_pydict(columns, schema=schema)
    pq.write_table(table, path, compression="zstd")


def git_head() -> str:
    """Return the current git HEAD commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return result.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_readme(path: Path, known_issues: list[str]) -> None:
    """Write the README. Only the Known Issues section varies between runs."""
    if known_issues:
        issues_md = "\n".join(f"- {line}" for line in known_issues)
    else:
        issues_md = "None recorded for the most recent run."

    text = f"""# Lab-processed Raman spectra

## What this folder is

Processed Raman spectra of iron-based ionic liquid reference compounds, derived from raw WITec acquisitions in `data/raw_lab/`. The contents here are the reference inputs for downstream library matching against Mars SuperCam spectra.

## Source of the raw data

Raw acquisitions live in `data/raw_lab/`, organized by compound directory. Each acquisition is loaded by `scripts/01_inspect_witec.py::load_acquisition`, which handles both the split export format (separate X-Axis, Y-Axis, Header, Information files) and the combined format (single Spec.Data file with header inline).

## Processing recipe

1. Cosmic ray removal. Compute the discrete second derivative of intensity. Flag points whose deviation from the median exceeds 8 * MAD of the second derivative. Replace flagged points with the median of a 5-point neighborhood. Iterate up to 3 times.
2. ALS baseline correction. Asymmetric least squares with `lam = 1e5`, `p = 0.001`, `niter = 10`. Reference: Eilers, P.H.C., Boelens, H.F.M. (2005), "Baseline correction with asymmetric least squares smoothing".
3. Trim to `150 <= wavenumber <= 1700 cm-1`.
4. Resample to a uniform 1 cm-1 grid (`np.arange(150, 1701, 1.0)`, 1551 points) by linear interpolation. Positions outside the native range are set to NaN; no extrapolation.
5. Normalize by dividing the resampled spectrum by its NaN-ignoring maximum.

## What was not done

- No smoothing.
- No SuperCam-resolution convolution.
- No replicate averaging.
- No outlier rejection across acquisitions.
- No condition labeling.

## File conventions

CSVs are written to `spectra/` with names of the form:

    {{compound}}_{{laser_nm}}nm_{{power_mw}}mW_spec{{NN}}.csv

`NN` is the WITec spectrum number, zero-padded to a minimum width of 2; existing wider numbers (e.g. `019`) are preserved.

CSV columns:

| column | description |
|---|---|
| `wavenumber_cm_inv` | Raman shift in cm^-1 on the uniform 150-1700 grid, two decimal places |
| `intensity_baseline_corrected` | After cosmic ray removal, ALS subtraction, and resample. NaN where the native range did not cover the grid point. |
| `intensity_normalized` | `intensity_baseline_corrected` divided by its NaN-ignoring max |

The Parquet file `lab_processed.parquet` contains one row per (acquisition, wavenumber) with this schema:

| column | dtype | source |
|---|---|---|
| `compound` | string | from directory map |
| `laser_nm` | int32 | from folder name or metadata |
| `power_mw` | int32 | from folder name or metadata |
| `spectrum_number` | string | from Acquisition |
| `source_folder` | string | relative to repo root |
| `sample_name` | string | from Acquisition |
| `sample_inferred` | bool | from Acquisition |
| `wavenumber_cm_inv` | float64 | uniform 150-1700, 1 cm^-1 spacing |
| `intensity_baseline_corrected` | float64 | post step 4 |
| `intensity_normalized` | float64 | post step 5 |
| `out_of_range_low` | bool | true if the acquisition's native range starts above 150 |
| `out_of_range_high` | bool | true if the acquisition's native range ends below 1700 |

The Parquet is canonical. CSVs are derived from the same in-memory arrays at write time. If they ever disagree, treat the Parquet as authoritative.

## Compound abbreviations

| canonical name | formula |
|---|---|
| EMIM-FeCl4 | [EMIM][FeCl4] |
| EMIM-FeBr4 | [EMIM][FeBr4] |
| EMIM2-Fe2Cl7 | [EMIM]2[Fe2Cl7] |
| EMIM2-FeSO4 | [EMIM]2[FeSO4] |

## Provenance

Provenance for each script run is written to `.provenance.json` (not tracked in git). It records the script version, git commit hash, generation timestamp, and any warnings raised. Re-running the script regenerates this file.

## Known issues

{issues_md}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    raw_lab = REPO_ROOT / "data" / "raw_lab"
    out_root = REPO_ROOT / "data" / "lab_processed"

    if not raw_lab.is_dir():
        raise FileNotFoundError(f"Missing raw lab directory: {raw_lab}")

    folders = discover_acquisition_folders(raw_lab)
    print(f"Discovered {len(folders)} acquisition folder(s):")
    by_compound: dict[str, int] = {}
    for f in folders:
        rel = f.relative_to(REPO_ROOT)
        compound_dir = f.relative_to(raw_lab).parts[0]
        tag = COMPOUND_MAP.get(compound_dir, "<unknown>")
        by_compound[tag] = by_compound.get(tag, 0) + 1
        print(f"  {rel}  ({tag})")
    print("By compound: " + ", ".join(f"{k}={v}" for k, v in by_compound.items()))

    if out_root.exists():
        shutil.rmtree(out_root)
    spectra_dir = out_root / "spectra"
    spectra_dir.mkdir(parents=True)

    columns: dict[str, list] = {
        "compound": [],
        "laser_nm": [],
        "power_mw": [],
        "spectrum_number": [],
        "source_folder": [],
        "sample_name": [],
        "sample_inferred": [],
        "wavenumber_cm_inv": [],
        "intensity_baseline_corrected": [],
        "intensity_normalized": [],
        "out_of_range_low": [],
        "out_of_range_high": [],
    }
    seen_csv_names: set[str] = set()
    warnings_log: list[str] = []
    n_spectra = 0

    for folder in folders:
        rel_parts = folder.relative_to(raw_lab).parts
        compound_dir = rel_parts[0]
        if compound_dir not in COMPOUND_MAP:
            raise RuntimeError(
                f"Unknown compound directory under data/raw_lab/: {compound_dir}"
            )
        compound = COMPOUND_MAP[compound_dir]
        is_compound_root = len(rel_parts) == 1

        folder_nm: int | None = None
        folder_mw: int | None = None
        if not is_compound_root:
            folder_nm, folder_mw = extract_laser_power_from_name(folder.name)
            if folder_nm is None or folder_mw is None:
                msg = (
                    f"Skipping {folder.relative_to(REPO_ROOT)}: "
                    "could not parse nm/mW from folder name"
                )
                warnings_log.append(msg)
                print(f"WARNING: {msg}")
                continue

        try:
            acquisitions = load_acquisition(folder)
        except Exception as exc:
            msg = (
                f"Skipping {folder.relative_to(REPO_ROOT)}: load failed "
                f"({type(exc).__name__}: {exc})"
            )
            warnings_log.append(msg)
            print(f"WARNING: {msg}")
            continue

        for acq in acquisitions:
            if is_compound_root:
                ex = lookup(acq.metadata, "Excitation Wavelength")
                lp = lookup(acq.metadata, "Laser Power")
                if not ex or not lp:
                    msg = (
                        f"Skipping {folder.name}/spec{acq.spectrum_number}: "
                        "metadata missing Excitation Wavelength or Laser Power"
                    )
                    warnings_log.append(msg)
                    print(f"WARNING: {msg}")
                    continue
                laser_nm = int(round(float(ex)))
                power_mw = int(round(float(lp)))
            else:
                assert folder_nm is not None and folder_mw is not None
                laser_nm = folder_nm
                power_mw = folder_mw
                ex = lookup(acq.metadata, "Excitation Wavelength")
                lp = lookup(acq.metadata, "Laser Power")
                if ex and lp:
                    meta_nm = int(round(float(ex)))
                    meta_mw = int(round(float(lp)))
                    if meta_nm != laser_nm or meta_mw != power_mw:
                        msg = (
                            f"{folder.relative_to(REPO_ROOT)} "
                            f"spec{acq.spectrum_number}: folder name says "
                            f"{laser_nm}nm/{power_mw}mW but metadata says "
                            f"{meta_nm}nm/{meta_mw}mW; using folder name"
                        )
                        warnings_log.append(msg)
                        print(f"WARNING: {msg}")

            spec_label = acq.spectrum_number.zfill(2)
            csv_name = (
                f"{compound}_{laser_nm}nm_{power_mw}mW_spec{spec_label}.csv"
            )
            if csv_name in seen_csv_names:
                raise RuntimeError(
                    f"Filename collision for {csv_name} (most recent source: "
                    f"{folder.relative_to(REPO_ROOT)})"
                )
            seen_csv_names.add(csv_name)

            y_corr_grid, y_norm_grid, oor_low, oor_high = process_acquisition(acq)
            write_csv(spectra_dir / csv_name, LAB_GRID, y_corr_grid, y_norm_grid)

            n_pts = LAB_GRID.size
            columns["compound"].extend([compound] * n_pts)
            columns["laser_nm"].extend([laser_nm] * n_pts)
            columns["power_mw"].extend([power_mw] * n_pts)
            columns["spectrum_number"].extend([acq.spectrum_number] * n_pts)
            columns["source_folder"].extend(
                [str(folder.relative_to(REPO_ROOT))] * n_pts
            )
            columns["sample_name"].extend([acq.sample_name] * n_pts)
            columns["sample_inferred"].extend([acq.sample_inferred] * n_pts)
            columns["wavenumber_cm_inv"].extend(LAB_GRID.tolist())
            columns["intensity_baseline_corrected"].extend(y_corr_grid.tolist())
            columns["intensity_normalized"].extend(y_norm_grid.tolist())
            columns["out_of_range_low"].extend([oor_low] * n_pts)
            columns["out_of_range_high"].extend([oor_high] * n_pts)

            n_spectra += 1
            note = ""
            if oor_low or oor_high:
                note = "  (resample NaNs at "
                note += "low" if oor_low else ""
                note += " & " if (oor_low and oor_high) else ""
                note += "high" if oor_high else ""
                note += " edge)"
            print(f"  processed {csv_name}  (n_points={n_pts}){note}")

    parquet_path = out_root / "lab_processed.parquet"
    write_parquet(columns, parquet_path)
    write_readme(out_root / "README.md", warnings_log)

    provenance = {
        "script": "scripts/02_process_witec.py",
        "git_commit_hash": git_head(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_folders_discovered": len(folders),
        "n_spectra_processed": n_spectra,
        "warnings": warnings_log,
    }
    (out_root / ".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    parquet_size = parquet_path.stat().st_size
    print()
    print(
        f"Wrote {n_spectra} spectra ({len(seen_csv_names)} CSVs) and "
        f"lab_processed.parquet ({parquet_size / 1024:.1f} KiB)."
    )
    if warnings_log:
        print(
            f"{len(warnings_log)} warning(s) recorded in README and provenance."
        )


if __name__ == "__main__":
    main()
