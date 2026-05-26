"""Self-match positive control for the matcher primitives in _processing.py.

Feeds the four convolved lab reference spectra back through the SNR and
cosine primitives in _processing.py (compute_local_polynomial_detrending,
compute_peak_quality_flags, compute_cosine_in_diagnostic_windows) and
reports per-(spectrum compound x template compound) numbers across 16
cells: 4 self-match (diagonal) and 12 cross-compound (off-diagonal).

Hard gates apply only to self-match cells, and only on properties that
are mathematically or structurally guaranteed if the matcher works:

  cosine >= 0.999   identical input through identical detrending must
                    produce identical residuals; cosine of a vector with
                    itself is exactly 1.0. Floor at 0.999 catches a
                    real bug if cosine drops meaningfully below 1.0.
  A-primary curvature_sign == -1 for all peaks
                    a clean lab peak has real negative curvature; a
                    parabola fit returning +1 or 0 indicates the fit
                    is broken.
  zero coverage gaps
                    the spectrum's own lab data must cover every Class A
                    catalog peak position. A skipped peak in a self-match
                    cell means the lab spectrum doesn't have data where
                    its own catalog says it should.

All other numbers (self-match SNR, position_offset, all cross-compound
quantities) are reported informationally without pass/fail gating. The
original aspirational gates on those quantities encoded incorrect priors:
self-match SNR runs in single digits (not hundreds) because polynomial-2
detrending leaves residuals at 5-15% of peak height; position_offset has
a ~0.1-0.5 quantization noise floor from the integer wavenumber grid
combined with fractional catalog centers; cross-compound cosine can run
high in single-window regimes (notably FeSO4) where polynomial residuals
align across compounds without any actual peak-shape match. See the
session log entry for commit D-part-2 follow-up for the full discussion.

Aggregate PASS when all 4 self-match cells pass the three hard gates.
Cross-compound cells contribute numbers but no gate. Exit 0 on PASS,
1 on FAIL. No parquet output, no results/ writes; the printed report
is the artifact.

References:
  See docs/figures/d_design/D_handoff_v4.md "Validation requirements"
  item 3 for the original handoff text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
CATALOG_PATH = (
    REPO_ROOT / "data" / "reference_library" / "peak_catalog.parquet"
)
REF_SPECTRA_PATH = (
    REPO_ROOT / "data" / "reference_library" / "reference_spectra.parquet"
)

COMPOUND_ORDER = (
    "EMIM-FeCl4",
    "EMIM-FeBr4",
    "EMIM2-Fe2Cl7",
    "EMIM2-FeSO4",
)

# Per-rule tolerance overrides keyed by (compound, rounded center).
# Mirrors 06 and 07. The four FeSO4 doublet rules at 1022.01, 1027.38,
# 1086.60, 1091.81 cm-1 use +/-2.5 cm-1 instead of the 5.0 default.
TOLERANCE_OVERRIDES: dict[tuple[str, float], float] = {
    ("EMIM2-FeSO4", 1022.01): 2.5,
    ("EMIM2-FeSO4", 1027.38): 2.5,
    ("EMIM2-FeSO4", 1086.60): 2.5,
    ("EMIM2-FeSO4", 1091.81): 2.5,
}
DEFAULT_TOLERANCE = 5.0

# Match the design constants from 06 so the validation runs under the
# production matcher rules.
POLY_DEGREE = 2
HALF_WINDOW = 30.0
PEER_TOL = 5.0
COSINE_HALF_WINDOW = 15.0

# Self-match cosine hard gate. Mathematically the cosine of a vector
# with itself is 1.0; same input through identical detrending produces
# identical residuals. Floor at 0.999 catches a real bug if cosine drops
# meaningfully below 1.0; numerical-precision drift below that is fine.
SELF_MATCH_COSINE_FLOOR = 0.999


def tolerance_for(compound: str, center: float) -> float:
    for (cmp, ctr), v in TOLERANCE_OVERRIDES.items():
        if cmp == compound and abs(ctr - center) < 0.05:
            return v
    return DEFAULT_TOLERANCE


def load_catalog() -> pd.DataFrame:
    """Load the 46 aggregate-convolved catalog rows with tolerance_used and
    peak_subclass columns added (mirrors 06 and 07).
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
    return df.sort_values(["compound", "position_cm_inv"]).reset_index(
        drop=True
    )


def load_reference_means(
    compound_order: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-compound mean of intensity_normalized_convolved across acquisitions
    flagged usable_for_supercam=True. Returns {compound: (wavenumbers, mean)}.
    Mirrors 06.
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
    catalog: pd.DataFrame,
    target_center: float,
    half_window: float = HALF_WINDOW,
) -> list[float]:
    """Catalog centers whose tolerance window overlaps [target_center +/-
    half_window], excluding the target itself. Cross-compound; uses each
    peer's own tolerance_used for the overlap check. Mirrors 06.
    """
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


def short_label(compound: str) -> str:
    """Strip the EMIM / EMIM2 prefix for compact report headers."""
    if compound.startswith("EMIM2-"):
        return compound[6:]
    if compound.startswith("EMIM-"):
        return compound[5:]
    return compound


def score_cell(
    spectrum_compound: str,
    template_compound: str,
    ref_means: dict[str, tuple[np.ndarray, np.ndarray]],
    catalog: pd.DataFrame,
) -> dict:
    """Score one (spectrum_compound, template_compound) cell.

    Treats the spectrum compound's lab data as the "Mars" argument and the
    template compound's catalog peaks as the targets. SNR and quality
    flags are computed for every Class A peak (primary + secondary) of
    the template; cosine is computed over the template's A-peak diagnostic
    windows with cross-compound peer exclusion.

    Returns a dict carrying the per-cell numbers used by both the gate
    logic and the printed report.
    """
    spec_wn, spec_intens = ref_means[spectrum_compound]
    template_wn, template_intens = ref_means[template_compound]
    if not np.array_equal(spec_wn, template_wn):
        raise RuntimeError(
            f"wavenumber grid mismatch between {spectrum_compound} and "
            f"{template_compound}"
        )

    template_a = catalog[
        (catalog["compound"] == template_compound)
        & (catalog["class_label"] == "A")
    ].sort_values("position_cm_inv")
    template_a_primary = template_a[template_a["peak_subclass"] == "primary"]
    template_a_secondary = template_a[
        template_a["peak_subclass"] == "secondary"
    ]
    all_catalog_centers = [float(c) for c in catalog["position_cm_inv"]]

    # Per-peak SNR plus quality. The SNR primitive raises ValueError when
    # the spectrum has no finite samples in the fit window or no grid
    # points in the tolerance window. EMIM-FeCl4 lab is NaN below 227 cm-1
    # (raw WiTec acquisitions did not extend that low), so cross-compound
    # scoring at FeBr4 218 or Fe2Cl7 191 from FeCl4 lab cannot produce a
    # number. Skip those peaks rather than crash; min/max aggregates run
    # over the peaks that did score, and the skip count is reported.
    primary_snrs: list[float] = []
    primary_offsets: list[float] = []
    primary_curvatures: list[int] = []
    n_primary_skipped = 0
    for _, peak in template_a_primary.iterrows():
        ctr = float(peak["position_cm_inv"])
        tol = float(peak["tolerance_used"])
        peers = build_peer_centers_for_window(catalog, ctr)
        try:
            result = compute_local_polynomial_detrending(
                spec_wn,
                spec_intens,
                ctr,
                tol,
                peers,
                half_window=HALF_WINDOW,
                peer_tol=PEER_TOL,
                poly_degree=POLY_DEGREE,
            )
        except ValueError:
            n_primary_skipped += 1
            continue
        flags = compute_peak_quality_flags(
            result["detrended_window"],
            result["wavenumbers_window"],
            result["peak_position"],
            ctr,
            tol,
        )
        primary_snrs.append(float(result["snr"]))
        primary_offsets.append(float(flags["position_offset_normalized"]))
        primary_curvatures.append(int(flags["peak_curvature_sign"]))

    secondary_snrs: list[float] = []
    n_secondary_skipped = 0
    for _, peak in template_a_secondary.iterrows():
        ctr = float(peak["position_cm_inv"])
        tol = float(peak["tolerance_used"])
        peers = build_peer_centers_for_window(catalog, ctr)
        try:
            result = compute_local_polynomial_detrending(
                spec_wn,
                spec_intens,
                ctr,
                tol,
                peers,
                half_window=HALF_WINDOW,
                peer_tol=PEER_TOL,
                poly_degree=POLY_DEGREE,
            )
        except ValueError:
            n_secondary_skipped += 1
            continue
        secondary_snrs.append(float(result["snr"]))

    template_window_centers = sorted(
        float(p) for p in template_a["position_cm_inv"]
    )
    cosine_val = compute_cosine_in_diagnostic_windows(
        spec_intens,
        template_intens,
        spec_wn,
        template_window_centers,
        peer_peak_centers=all_catalog_centers,
        tolerance=DEFAULT_TOLERANCE,
        half_window=COSINE_HALF_WINDOW,
        peer_tol=PEER_TOL,
        poly_degree=POLY_DEGREE,
    )

    return {
        "spectrum_compound": spectrum_compound,
        "template_compound": template_compound,
        "n_a_primary": len(primary_snrs),
        "n_a_secondary": len(secondary_snrs),
        "n_a_primary_skipped": n_primary_skipped,
        "n_a_secondary_skipped": n_secondary_skipped,
        "n_a_primary_catalog": len(template_a_primary),
        "n_a_secondary_catalog": len(template_a_secondary),
        "min_a_primary_snr": (
            float(min(primary_snrs)) if primary_snrs else float("nan")
        ),
        "max_a_primary_snr": (
            float(max(primary_snrs)) if primary_snrs else float("nan")
        ),
        "min_a_secondary_snr": (
            float(min(secondary_snrs)) if secondary_snrs else float("nan")
        ),
        "max_a_secondary_snr": (
            float(max(secondary_snrs)) if secondary_snrs else float("nan")
        ),
        "max_position_offset_a_primary": (
            float(max(primary_offsets))
            if primary_offsets
            else float("nan")
        ),
        "all_a_primary_curvature_minus_one": all(
            c == -1 for c in primary_curvatures
        ),
        "primary_curvature_signs": tuple(primary_curvatures),
        "cosine": float(cosine_val),
    }


def evaluate_gates(cell: dict) -> tuple[bool, list[str]]:
    """Return (cell_pass, gate_lines).

    gate_lines is a list of one-line strings suitable for direct printing
    in the report. Each line is prefixed with PASS, FAIL, or SKIP.
    """
    spec = cell["spectrum_compound"]
    tmpl = cell["template_compound"]
    is_self = spec == tmpl
    lines: list[str] = []
    cell_pass = True

    if is_self:
        if cell["n_a_primary"] == 0:
            lines.append(
                f"  FAIL  no A-primary peaks for {tmpl} "
                f"(catalog topology error)"
            )
            return False, lines

        cos = cell["cosine"]
        ok = cos >= SELF_MATCH_COSINE_FLOOR
        lines.append(
            f"  {'PASS' if ok else 'FAIL'}  self-match cosine "
            f"{cos:.4f} >= {SELF_MATCH_COSINE_FLOOR} floor"
        )
        cell_pass = cell_pass and ok

        ok = cell["all_a_primary_curvature_minus_one"]
        sigs = list(cell["primary_curvature_signs"])
        lines.append(
            f"  {'PASS' if ok else 'FAIL'}  self-match A-primary "
            f"curvature_sign all -1, observed {sigs}"
        )
        cell_pass = cell_pass and ok

        n_prim_skip = cell["n_a_primary_skipped"]
        n_sec_skip = cell["n_a_secondary_skipped"]
        ok = (n_prim_skip == 0) and (n_sec_skip == 0)
        lines.append(
            f"  {'PASS' if ok else 'FAIL'}  self-match coverage: "
            f"{n_prim_skip} primary and {n_sec_skip} secondary peaks "
            f"skipped (zero expected)"
        )
        cell_pass = cell_pass and ok
    # Cross-compound cells: no hard gates. The numbers print in the cell
    # numbers block above; nothing additional to gate here.

    return cell_pass, lines


def print_report(cells: list[dict]) -> bool:
    print("Self-match validation report")
    print("=" * 78)
    print()
    print("Self-match hard gates (per diagonal cell):")
    print(f"  cosine >= {SELF_MATCH_COSINE_FLOOR}")
    print("  A-primary curvature_sign == -1 for all peaks")
    print("  zero coverage gaps (all catalog peaks scored)")
    print()
    print(
        "Other numbers (SNR, position_offset, cross-cosine) are reported "
        "for inspection without pass/fail gating."
    )
    print()

    n_self = sum(
        1 for c in cells if c["spectrum_compound"] == c["template_compound"]
    )
    n_cross = len(cells) - n_self
    n_self_pass = 0
    for cell in cells:
        spec = cell["spectrum_compound"]
        tmpl = cell["template_compound"]
        is_self = spec == tmpl
        kind = "self" if is_self else "cross"
        header = (
            f"[{kind}] {short_label(spec)} spectrum "
            f"vs {short_label(tmpl)} template"
        )
        print(header)
        print("-" * len(header))
        n_prim_skip = cell["n_a_primary_skipped"]
        n_sec_skip = cell["n_a_secondary_skipped"]
        print(
            f"  n_a_primary={cell['n_a_primary']}/"
            f"{cell['n_a_primary_catalog']} scored, "
            f"n_a_secondary={cell['n_a_secondary']}/"
            f"{cell['n_a_secondary_catalog']} scored"
        )
        if n_prim_skip > 0 or n_sec_skip > 0:
            print(
                f"  skipped due to NaN coverage gap: "
                f"{n_prim_skip} primary, {n_sec_skip} secondary"
            )
        if cell["n_a_primary"] > 0:
            print(
                f"  max_a_primary_snr   = {cell['max_a_primary_snr']:.2f}, "
                f"min = {cell['min_a_primary_snr']:.2f}"
            )
            print(
                f"  max_position_offset = "
                f"{cell['max_position_offset_a_primary']:.4f}"
            )
            print(
                f"  curvature_signs     = "
                f"{list(cell['primary_curvature_signs'])}"
            )
        if cell["n_a_secondary"] > 0:
            print(
                f"  max_a_secondary_snr = {cell['max_a_secondary_snr']:.2f}, "
                f"min = {cell['min_a_secondary_snr']:.2f}"
            )
        print(f"  cosine              = {cell['cosine']:.4f}")
        cell_pass, gate_lines = evaluate_gates(cell)
        for line in gate_lines:
            print(line)
        if is_self and cell_pass:
            n_self_pass += 1
        print()

    print("=" * 78)
    overall = "PASS" if n_self_pass == n_self else "FAIL"
    print(
        f"Aggregate: {n_self_pass} of {n_self} self-match cells passed "
        f"hard gates. {n_cross} cross-match cells reported "
        f"(no gates applied). -> {overall}"
    )
    return n_self_pass == n_self


def main() -> int:
    if not CATALOG_PATH.is_file():
        raise FileNotFoundError(f"Missing input: {CATALOG_PATH}")
    if not REF_SPECTRA_PATH.is_file():
        raise FileNotFoundError(f"Missing input: {REF_SPECTRA_PATH}")

    catalog = load_catalog()
    if len(catalog) != 46:
        raise ValueError(
            f"expected 46 aggregate-convolved catalog rows, got "
            f"{len(catalog)}"
        )
    ref_means = load_reference_means(COMPOUND_ORDER)

    cells: list[dict] = []
    for spectrum_compound in COMPOUND_ORDER:
        for template_compound in COMPOUND_ORDER:
            cell = score_cell(
                spectrum_compound, template_compound, ref_means, catalog
            )
            cells.append(cell)

    overall_pass = print_report(cells)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
